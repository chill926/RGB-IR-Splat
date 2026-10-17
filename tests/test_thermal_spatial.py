import unittest
import torch
from utils.thermal_spatial import TemperatureSpatialPrior


class SpatialPriorTest(unittest.TestCase):
    def build(self, points=None, labels=None, colors=None, support=None, opacity=None, **options):
        points = torch.tensor([[0.,0,0],[.01,0,0],[1.,0,0],[1.01,0,0]]) if points is None else points
        count = len(points)
        return TemperatureSpatialPrior.build(points, torch.ones(count) if opacity is None else opacity,
            torch.zeros(count,3) if colors is None else colors,
            torch.zeros(count,dtype=torch.long) if labels is None else labels,
            torch.ones(count,dtype=torch.bool) if support is None else support, neighbors=1, **options)

    def test_constant_temperature_has_zero_loss_and_gradient(self):
        prior=self.build()
        temperature=torch.full((4,1),300.,requires_grad=True)
        loss=prior(temperature);loss.backward()
        self.assertEqual(float(loss),0.)
        self.assertEqual(float(temperature.grad.abs().max()),0.)

    def test_gradient_suppresses_local_spike_without_joining_distant_clusters(self):
        prior=self.build()
        temperature=torch.tensor([[300.],[320.],[290.],[290.]],requires_grad=True)
        prior(temperature).backward()
        self.assertLess(float(temperature.grad[0]),0.)
        self.assertGreater(float(temperature.grad[1]),0.)
        self.assertEqual(float(temperature.grad[2]),0.)
        self.assertEqual(float(temperature.grad[3]),0.)

    def test_material_boundary_and_unknown_edges_are_attenuated(self):
        for labels in [torch.tensor([0,0,1,2]),torch.tensor([0,0,-1,-1])]:
            prior=self.build(labels=labels)
            self.assertAlmostEqual(float(prior.weights[1]/prior.weights[0]),.1,places=5)

    def test_RGB_boundary_does_not_force_two_regions_to_equal_temperature(self):
        colors=torch.tensor([[0.,0,0],[0.,0,0],[0.,0,0],[1.,1,1]])
        prior=self.build(colors=colors)
        pairs=list(zip(prior.source.tolist(),prior.target.tolist()))
        self.assertEqual(pairs,[(0,1)])

    def test_unseen_or_zero_opacity_Gaussians_do_not_form_edges(self):
        for kwargs in [{'support':torch.tensor([True,True,False,True])},
                       {'opacity':torch.tensor([1.,1.,0.,1.])}]:
            prior=self.build(**kwargs)
            self.assertNotIn(2,prior.source.tolist()+prior.target.tolist())

    def test_long_isolated_connections_removed(self):
        points=torch.tensor([[0.,0,0],[.01,0,0],[.02,0,0],[100.,0,0]])
        prior=self.build(points=points)
        self.assertNotIn(3,prior.source.tolist()+prior.target.tolist())
        self.assertGreater(prior.protocol['distance_filtered_edges'],0)

    def test_graph_is_frozen_and_reproducible(self):
        a,b=self.build(),self.build()
        self.assertEqual(a.protocol['graph_sha256'],b.protocol['graph_sha256'])
        self.assertFalse(any(value.requires_grad for value in a.buffers()))

    def test_invalid_scale_and_edge_weights_rejected(self):
        for kwargs in [{'temperature_scale_K':0},{'unknown_weight':-1},{'rgb_sigma':float('nan')}]:
            with self.assertRaises(ValueError):self.build(**kwargs)


if __name__=='__main__':unittest.main()
