"""Independent analytic thermal scene, calibrated observations, and public-only IO.

Generation uses NumPy geometry intersections and Planck integration, not the
inverse model. The truth/ directory is read only by explicit Oracle/evaluation.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blackbody(temperature, samples=96):
    """Average spectral radiance over uniform 8-14 um response, W/m^3/sr."""
    t = np.asarray(temperature, dtype=np.float64)
    if np.any(t <= 0) or not np.isfinite(t).all():
        raise ValueError("Temperature must be finite Kelvin > 0")
    wavelength = np.linspace(8e-6, 14e-6, samples)
    h, c, kb = 6.62607015e-34, 299792458., 1.380649e-23
    spectral = 2 * h * c * c / wavelength ** 5 / np.expm1(h * c / (wavelength * kb * t[..., None]))
    return np.trapz(spectral, wavelength, axis=-1) / (wavelength[-1] - wavelength[0])


def sh_basis(normals):
    n = np.asarray(normals)
    x, y, z = np.moveaxis(n, -1, 0)
    c1 = .4886025119029199
    c2 = [1.0925484305920792, -1.0925484305920792, .31539156525252005,
          -1.0925484305920792, .5462742152960396]
    return np.stack((-c1*y, c1*z, -c1*x, c2[0]*x*y, c2[1]*y*z,
        c2[2]*(2*z*z-x*x-y*y), c2[3]*x*z, c2[4]*(x*x-y*y)), axis=-1)


def environment(normals, mean, illumination_coefficients):
    # E(n)=(1/pi) integral_hemisphere L_env(w) max(n.w,0) dw.
    # Diffuse convolution attenuates SH bands 1/2 by 2/3 and 1/4.
    attenuation = np.array([2/3]*3 + [.25]*5)
    return mean + sh_basis(normals) @ (np.asarray(illumination_coefficients) * attenuation)


def forward_radiance(temperature, epsilon0, k, environment_radiance, response):
    epsilon = np.asarray(epsilon0) + np.asarray(k) * (np.asarray(temperature) - 300.)
    if np.any((epsilon <= .01) | (epsilon >= .99)):
        raise ValueError("Synthetic emissivity leaves (0.01,0.99); change scene parameters")
    return epsilon * response(temperature) + (1-epsilon) * environment_radiance, epsilon


def scene_definition(k_scale=1., temperature_span=60., environment_temperature=295., directional=.15):
    mean = float(blackbody(environment_temperature))
    coefficients = mean * directional * np.array([.3, -.2, .4, .1, 0., -.2, .05, .1])
    return {
        "materials": [
            {"name":"generic_high_epsilon", "epsilon0":.9, "k":-1e-4*k_scale,"rgb":[.75,.25,.15]},
            {"name":"generic_low_epsilon", "epsilon0":.35, "k":2e-4*k_scale,"rgb":[.2,.55,.75]},
            {"name":"generic_medium_epsilon", "epsilon0":.65, "k":1e-4*k_scale,"rgb":[.7,.65,.2]}],
        "objects": [
            {"kind":"panel","center":[-.7,0.,0.],"width":1.2,"height":1.6,"angle":.1,"material":0},
            {"kind":"panel","center":[.65,0.,-.15],"width":1.1,"height":1.5,"angle":-.4,"material":1},
            {"kind":"sphere","center":[0.,-.1,.45],"radius":.34,"material":2}],
        "temperature_span_K":float(temperature_span), "temperature_reference_K":300.,
        "environment_mean":mean,"illumination_SH_1_2":coefficients.tolist(),
        "environment_temperature_K":float(environment_temperature),
        "note":"Controlled synthetic coefficients, not measured material constants"}


def panel_axes(obj):
    a = obj["angle"]
    return np.array([math.cos(a),0.,-math.sin(a)]), np.array([0.,1.,0.]), np.array([math.sin(a),0.,math.cos(a)])


def surface_temperature(obj, points, normals, span):
    if obj["kind"] == "panel":
        u, v, _ = panel_axes(obj)
        relative = points - np.asarray(obj["center"])
        x, y = relative @ u / obj["width"], relative @ v / obj["height"]
        return 310. + span * (.55*x + .3*y + .12*np.sin(2*np.pi*x)*np.cos(np.pi*y))
    return 310. + span * (.3*normals[..., 2] + .15*normals[..., 1])


def intersect(scene, origins, directions):
    """Nearest opaque analytic surface along each ray; no thermal parameters used."""
    shape = directions.shape[:-1]
    origins = np.broadcast_to(origins, directions.shape)
    distance = np.full(shape, np.inf)
    ids = np.full(shape, -1, dtype=np.int32)
    normals = np.zeros_like(directions)
    for index, obj in enumerate(scene["objects"]):
        center = np.asarray(obj["center"])
        if obj["kind"] == "sphere":
            q = origins-center; b = np.sum(q*directions,axis=-1)
            discriminant = b*b-np.sum(q*q,axis=-1)+obj["radius"]**2
            t = -b-np.sqrt(np.maximum(discriminant,0))
            valid = (discriminant>=0)&(t>1e-5)
            n = (origins+t[...,None]*directions-center)/obj["radius"]
        else:
            u,v,n0 = panel_axes(obj)
            denominator = directions @ n0
            t = ((center-origins) @ n0)/np.where(np.abs(denominator)>1e-10,denominator,1e-10)
            hit = origins+t[...,None]*directions-center
            valid = (denominator < -1e-8)&(t>1e-5)&(np.abs(hit@u)<=obj["width"]/2)&(np.abs(hit@v)<=obj["height"]/2)
            n = np.broadcast_to(n0,directions.shape)
        take = valid & (t<distance)
        distance[take],ids[take],normals[take] = t[take],index,n[take]
    finite = np.where(np.isfinite(distance),distance,0)
    return origins+finite[...,None]*directions,normals,ids,distance


def make_camera(index, views, width, height):
    azimuth = math.radians(-55+110*index/(views-1))
    elevation = math.radians(10+8*math.sin(index*2*math.pi/views))
    target=np.array([0.,0.,.1]);center=target+3.5*np.array([math.sin(azimuth)*math.cos(elevation),math.sin(elevation),math.cos(azimuth)*math.cos(elevation)])
    forward=(target-center)/np.linalg.norm(target-center)
    right=np.cross(forward,[0.,1.,0.]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    rotation=np.stack((right,down,forward),axis=1)
    fovx=math.radians(55);fx=width/(2*math.tan(fovx/2))
    return {"name":f"view_{index:03d}","width":width,"height":height,"fx":fx,"fy":fx,
            "cx":width/2,"cy":height/2,"camera_to_world_rotation":rotation.tolist(),"center":center.tolist(),
            "split":"test" if index%5==0 else "validation" if index%5==1 else "fit"}


def camera_rays(camera):
    y,x=np.mgrid[:camera["height"],:camera["width"]]
    ray=np.stack(((x+.5-camera["cx"])/camera["fx"],(y+.5-camera["cy"])/camera["fy"],np.ones_like(x)),axis=-1)
    ray=ray@np.asarray(camera["camera_to_world_rotation"]).T
    return ray/np.linalg.norm(ray,axis=-1,keepdims=True)


def make_geometry(scene, grid, cameras):
    points=[];normals=[];ids=[];scales=[];areas=[]
    for object_id,obj in enumerate(scene["objects"]):
        if obj["kind"]=="panel":
            u,v,n=panel_axes(obj)
            x,y=np.meshgrid((np.arange(grid)+.5)/grid-.5,(np.arange(grid)+.5)/grid-.5)
            p=np.asarray(obj["center"])+x.reshape(-1,1)*obj["width"]*u+y.reshape(-1,1)*obj["height"]*v
            n=np.broadcast_to(n,p.shape);area=obj["width"]*obj["height"];spacing=math.sqrt(area/len(p))
        else:
            count=2*grid*grid;j=np.arange(count);z=1-2*(j+.5)/count;angle=j*math.pi*(3-math.sqrt(5))
            n=np.stack((np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z),axis=1)
            p=np.asarray(obj["center"])+obj["radius"]*n;area=4*math.pi*obj["radius"]**2;spacing=math.sqrt(area/count)
        points.append(p);normals.append(n);ids.append(np.full(len(p),object_id));scales.append(np.tile([spacing*.65,spacing*.65,spacing*.06],(len(p),1)));areas.append(np.full(len(p),area/len(p)))
    points=np.concatenate(points);normals=np.concatenate(normals);object_ids=np.concatenate(ids)
    material_ids=np.array([scene["objects"][i]["material"] for i in object_ids],dtype=np.int64)
    q=np.column_stack((1+normals[:,2],-normals[:,1],normals[:,0],np.zeros(len(points))))
    south=np.linalg.norm(q,axis=1)<1e-8;q[south]=[0,1,0,0];q/=np.linalg.norm(q,axis=1,keepdims=True)
    def visible(camera):
        direction=points-np.asarray(camera["center"]);length=np.linalg.norm(direction,axis=1);direction/=length[:,None]
        _,_,hit,distance=intersect(scene,np.asarray(camera["center"]),direction)
        local=(points-np.asarray(camera["center"]))@np.asarray(camera["camera_to_world_rotation"])
        z=local[:,2];x=camera["fx"]*local[:,0]/np.maximum(z,1e-9)+camera["cx"];y=camera["fy"]*local[:,1]/np.maximum(z,1e-9)+camera["cy"]
        return (hit==object_ids)&(np.abs(distance-length)<1e-4)&(z>0)&(x>=0)&(x<camera["width"])&(y>=0)&(y<camera["height"])
    supports={s:np.any([visible(c) for c in cameras if c["split"]==s],axis=0) for s in ("fit","validation","test")}
    rgb=np.asarray([scene["materials"][i]["rgb"] for i in material_ids])
    geometry={"xyz":points.astype(np.float32),"normals":normals.astype(np.float32),"scaling":np.concatenate(scales).astype(np.float32),
        "rotation":q.astype(np.float32),"opacity":np.full((len(points),1),.995,dtype=np.float32),"rgb":rgb.astype(np.float32),
        "material_ids":material_ids,"object_ids":object_ids,"area":np.concatenate(areas).astype(np.float32),"fit_support":supports["fit"]}
    return geometry,supports


def thermal_values(scene,points,normals,object_ids,response):
    temperature=np.zeros(object_ids.shape);epsilon0=np.zeros_like(temperature);k=np.zeros_like(temperature);material=np.full(object_ids.shape,-1,dtype=np.int32)
    for i,obj in enumerate(scene["objects"]):
        selected=object_ids==i
        temperature[selected]=surface_temperature(obj,points[selected],normals[selected],scene["temperature_span_K"])
        m=scene["materials"][obj["material"]];epsilon0[selected]=m["epsilon0"];k[selected]=m["k"];material[selected]=obj["material"]
    hit=object_ids>=0;E=environment(normals[hit],scene["environment_mean"],scene["illumination_SH_1_2"])
    radiance=np.zeros_like(temperature);epsilon=np.zeros_like(temperature);environment_map=np.zeros_like(temperature)
    radiance[hit],epsilon[hit]=forward_radiance(temperature[hit],epsilon0[hit],k[hit],E,response);environment_map[hit]=E
    return {"temperature_K":temperature,"epsilon":epsilon,"environment":environment_map,"radiance":radiance,"material":material}


def palette(normalized):
    f=np.clip((normalized-.06)/.24,0,1)
    return np.stack((np.clip(2*f,0,1),1-np.abs(2*f-1),np.clip(2*(1-f),0,1)),axis=-1)


def generate_dataset(output,width=320,height=240,views=40,grid=24,seed=2027,k_scale=1.,temperature_span=60.,
                     epsilon_bias=0.,noise_std_dn=0.,quantize=False,renderer="analytic",matched_renderer=None):
    output=Path(output).resolve()
    if output.exists():raise ValueError("Dataset already exists; choose a new output")
    if width<24 or height<24 or views<10 or grid<4 or not all(math.isfinite(v) for v in (k_scale,temperature_span,epsilon_bias,noise_std_dn)) or temperature_span<0 or noise_std_dn<0:
        raise ValueError("Invalid dataset dimensions/physical settings")
    if renderer not in ("analytic","gaussian_matched") or (renderer=="gaussian_matched" and matched_renderer is None):raise ValueError("Missing/invalid rendering backend")
    scene=scene_definition(k_scale,temperature_span)
    cameras=[make_camera(i,views,width,height) for i in range(views)]
    geometry,supports=make_geometry(scene,grid,cameras)
    knots=np.linspace(250,450,16384);table=blackbody(knots)
    def response(t):return np.interp(t,knots,table)
    truth=thermal_values(scene,geometry["xyz"],geometry["normals"],geometry["object_ids"],response)
    if np.any((truth["temperature_K"]<250)|(truth["temperature_K"]>450)):raise ValueError("Truth exceeds calibrated temperature range")
    gain=50000/(table[-1]-table[0]);offset=1000-gain*table[0]
    output.mkdir(parents=True);(output/"observations").mkdir();(output/"preview").mkdir();(output/"truth/frames").mkdir(parents=True)
    np.savez_compressed(output/"geometry.npz",**geometry)
    np.savez_compressed(output/"truth/gaussians.npz",**truth,**{s+"_visible":v for s,v in supports.items()},
                        k_by_material=np.array([m["k"] for m in scene["materials"]]),epsilon0_by_material=np.array([m["epsilon0"] for m in scene["materials"]]))
    (output/"truth/scene.json").write_text(json.dumps(scene,indent=2)+"\n")
    public={"kind":"physical_thermal_synthetic_v1","renderer":renderer,"geometry":"geometry.npz","geometry_sha256":file_hash(output/"geometry.npz"),
        "geometry_role":"known_frozen_surface_geometry_control","sensor":{"definition":"DN=gain*band_average_radiance+offset+noise",
        "gain":gain,"offset":offset,"radiance_min":float(table[0]),"radiance_span":float(table[-1]-table[0]),"noise_std_dn":noise_std_dn,"quantized":bool(quantize),
        "temperature_bounds_K":[250.,450.],"spectral_band_m":[8e-6,14e-6],"spectral_response":"uniform_normalized_to_unit_integral","radiance_units":"W m^-3 sr^-1"},
        "materials":[{"name":m["name"],"epsilon0_prior":float(np.clip(m["epsilon0"]+epsilon_bias,.02,.98)),"rgb":m["rgb"]} for m in scene["materials"]],
        "temp_ref_K":300.,"initial_temperature_K":300.,"seed":seed,"frames":[],
        "mask_policy":"foreground_eroded_one_pixel_per_object","background":"invalid_unmodeled_rays_not_temperature_measurements",
        "truth_policy":"truth_directory_excluded_from_main_training; explicit_Oracle_and_final_evaluation_only",
        "preview_mapping":"fixed_triangular_palette_of_normalized_radiance; no_png_inverse_for_training"}
    rng=np.random.default_rng(seed)
    for camera in cameras:
        rays=camera_rays(camera);points,normals,ids,depth=intersect(scene,np.asarray(camera["center"]),rays)
        values=thermal_values(scene,points,normals,ids,response)
        mask=np.zeros(ids.shape,dtype=bool)
        for i in range(len(scene["objects"])):mask |= binary_erosion(ids==i,iterations=1)
        radiance=values["radiance"]
        if renderer=="gaussian_matched":radiance=matched_renderer(geometry,truth,camera,public["sensor"])
        clean_dn=gain*radiance+offset
        observed=clean_dn+rng.normal(0,noise_std_dn,clean_dn.shape)
        saturated=(observed<0)|(observed>65535)
        if quantize:observed=np.clip(np.rint(observed),0,65535)
        if renderer=="analytic":observed[ids<0]=1000
        name=camera["name"]
        np.savez_compressed(output/"observations"/(name+".npz"),dn=observed.astype(np.float32),mask=mask,foreground=ids>=0)
        np.savez_compressed(output/"truth/frames"/(name+".npz"),**values,depth=depth.astype(np.float32),normals=normals.astype(np.float32),clean_observation_radiance=radiance.astype(np.float32))
        image=palette(((observed-offset)/gain-table[0])/(table[-1]-table[0]));image[ids<0]=0
        Image.fromarray(np.rint(image*255).astype(np.uint8)).save(output/"preview"/(name+"_ir.png"))
        rgb=np.zeros((*ids.shape,3))
        for i,obj in enumerate(scene["objects"]):rgb[ids==i]=scene["materials"][obj["material"]]["rgb"]
        Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(output/"preview"/(name+"_rgb.png"))
        path="observations/"+name+".npz"
        public["frames"].append(dict(camera,observation=path,sha256=file_hash(output/path),saturated_valid_fraction=float(saturated[mask].mean())))
    (output/"dataset.json").write_text(json.dumps(public,indent=2,allow_nan=False)+"\n")
    contact=Image.new("RGB",(width*4,height*2+40),"white");draw=ImageDraw.Draw(contact)
    selected=np.linspace(0,views-1,4,dtype=int)
    for column,i in enumerate(selected):
        camera=cameras[i]
        for row,modality in enumerate(("rgb","ir")):
            contact.paste(Image.open(output/"preview"/(camera["name"]+"_"+modality+".png")),(column*width,row*height+20))
        draw.text((column*width+4,3),camera["name"]+" "+camera["split"],fill="black")
    contact.save(output/"preview/contact.png")
    return public


def load_public(directory,splits=("fit","validation")):
    directory=Path(directory).resolve();document=json.loads((directory/"dataset.json").read_text())
    if document.get("kind")!="physical_thermal_synthetic_v1":raise ValueError("Unsupported synthetic dataset")
    if file_hash(directory/document["geometry"])!=document["geometry_sha256"]:raise ValueError("Geometry checksum mismatch")
    geometry=dict(np.load(directory/document["geometry"],allow_pickle=False))
    frames={s:[] for s in splits}
    for frame in document["frames"]:
        if frame["split"] not in frames:continue
        path=(directory/frame["observation"]).resolve()
        if directory not in path.parents or file_hash(path)!=frame["sha256"]:raise ValueError("Observation path/hash mismatch")
        observation=dict(np.load(path,allow_pickle=False))
        if not observation["mask"].any() or not np.isfinite(observation["dn"]).all():raise ValueError("Invalid observation")
        frames[frame["split"]].append((frame,observation))
    if any(not values for values in frames.values()):raise ValueError("Empty requested split")
    return document,geometry,frames
