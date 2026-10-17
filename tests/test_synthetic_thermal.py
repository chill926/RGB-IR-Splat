import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from utils.synthetic_thermal import (blackbody,environment,forward_radiance,scene_definition,
    make_camera,camera_rays,intersect,generate_dataset,load_public,file_hash)


class SyntheticPhysicsTest(unittest.TestCase):
    def test_planck_is_monotone_and_positive_in_Kelvin(self):
        values=blackbody(np.array([250.,300.,350.,450.]))
        self.assertTrue(np.all(values>0));self.assertTrue(np.all(np.diff(values)>0))
        with self.assertRaises(ValueError):blackbody(0.)

    def test_emission_reflection_balance_and_k_dependence(self):
        T=np.array([290.,320.]);E=np.array([4.,4.])
        L,epsilon=forward_radiance(T,.6,.001,E,lambda x:x)
        np.testing.assert_allclose(epsilon,[.59,.62])
        np.testing.assert_allclose(L,epsilon*T+(1-epsilon)*E)

    def test_invalid_emissivity_is_rejected_not_silently_clipped(self):
        with self.assertRaises(ValueError):forward_radiance(np.array([450.]),.9,.001,1.,lambda x:x)

    def test_thermal_equilibrium_is_blackbody_independent_of_emissivity(self):
        T=np.array([290.,320.]);B=blackbody(T)
        for epsilon0 in (.2,.8):
            L,_=forward_radiance(T,epsilon0,.0001,B,blackbody)
            np.testing.assert_allclose(L,B,rtol=1e-12)

    def test_isotropic_environment_is_constant(self):
        normals=np.eye(3)
        np.testing.assert_allclose(environment(normals,5.,np.zeros(8)),[5.,5.,5.])

    def test_diffuse_SH_convolution_matches_numerical_hemisphere_integral(self):
        from utils.synthetic_thermal import sh_basis
        n=100000;j=np.arange(n);z=1-2*(j+.5)/n;phi=j*np.pi*(3-np.sqrt(5))
        directions=np.stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z),axis=1)
        coeff=np.array([.1,-.2,.1,.05,.03,-.02,.01,.01])
        L=2.+sh_basis(directions)@coeff
        numeric=4*np.mean(L*np.maximum(z,0))
        analytic=environment(np.array([[0.,0.,1.]]),2.,coeff)[0]
        self.assertAlmostEqual(numeric,analytic,places=4)

    def test_camera_axes_and_rays_are_consistent(self):
        camera=make_camera(3,10,32,24);R=np.array(camera["camera_to_world_rotation"])
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R),1.)
        np.testing.assert_allclose(np.linalg.norm(camera_rays(camera),axis=-1),1.)

    def test_nearest_surface_occludes_far_panel(self):
        scene=scene_definition();_,_,ids,depth=intersect(scene,np.array([0.,-.1,3.]),np.array([[0.,0.,-1.]]))
        self.assertEqual(int(ids[0]),2);self.assertAlmostEqual(float(depth[0]),3.-.45-.34)

    def test_dataset_public_training_IO_excludes_truth_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"scene";generate_dataset(root,width=32,height=24,views=10,grid=6)
            raw_load=np.load
            paths=[]
            def guarded(path,*a,**kw):
                paths.append(str(path));self.assertNotIn("/truth/",str(path))
                return raw_load(path,*a,**kw)
            with patch("numpy.load",side_effect=guarded):
                document,geometry,frames=load_public(root)
            self.assertEqual(len(frames["fit"]),6);self.assertEqual(len(frames["validation"]),2)
            test_paths={str((root/f["observation"]).resolve()) for f in document["frames"] if f["split"]=="test"}
            self.assertTrue(test_paths.isdisjoint(paths));self.assertNotIn('"k":',(root/"dataset.json").read_text())
            self.assertTrue((root/"truth/gaussians.npz").exists())

    def test_generation_is_deterministic_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/"a",Path(directory)/"b"
            for p in (a,b):generate_dataset(p,width=32,height=24,views=10,grid=6,noise_std_dn=2.)
            self.assertEqual(file_hash(a/"observations/view_000.npz"),file_hash(b/"observations/view_000.npz"))
            with self.assertRaises(ValueError):generate_dataset(a)

    def test_zero_k_and_biased_prior_are_explicit_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"scene";d=generate_dataset(p,width=32,height=24,views=10,grid=6,k_scale=0.,epsilon_bias=.03)
            truth=dict(np.load(p/"truth/gaussians.npz",allow_pickle=False))
            np.testing.assert_allclose(truth["k_by_material"],0.)
            np.testing.assert_allclose([m["epsilon0_prior"] for m in d["materials"]],truth["epsilon0_by_material"]+.03)

    def test_matched_renderer_retains_its_coverage_outside_analytic_silhouette(self):
        def renderer(geometry,truth,camera,sensor):
            return np.full((camera["height"],camera["width"]),sensor["radiance_min"]+.01*sensor["radiance_span"])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"scene"
            generate_dataset(root,width=32,height=24,views=10,grid=6,renderer="gaussian_matched",matched_renderer=renderer)
            observation=dict(np.load(root/"observations/view_000.npz",allow_pickle=False))
            self.assertTrue((~observation["foreground"]).any())
            np.testing.assert_allclose(observation["dn"],1500.,rtol=1e-6)


if __name__=="__main__":unittest.main()
