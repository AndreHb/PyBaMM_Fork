#
# Test for the operator class
#
import pybamm
from tests import get_2p1d_mesh_for_testing, get_unit_2p1D_mesh_for_testing
import numpy as np
import unittest


@unittest.skipIf(pybamm.have_scikit_fem(), "scikit-fem not installed")
class TestScikitFiniteElement(unittest.TestCase):
    def test_not_implemented(self):
        mesh = get_2p1d_mesh_for_testing()
        spatial_method = pybamm.ScikitFiniteElement(mesh)
        self.assertEqual(spatial_method.mesh, mesh)
        with self.assertRaises(NotImplementedError):
            spatial_method.gradient(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.divergence(None, None, None)
        with self.assertRaises(NotImplementedError):
            spatial_method.indefinite_integral(None, None)

    def test_discretise_equations(self):
        # get mesh
        mesh = get_2p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.ScikitFiniteElement,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        # discretise some equations
        var = pybamm.Variable("var", domain="current collector")
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        disc.set_variable_slices([var])
        y_test = np.ones(mesh["current collector"][0].npts)
        unit_source = pybamm.Broadcast(1, "current collector")
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

        for eqn in [
            pybamm.laplacian(var),
            pybamm.source(unit_source, var),
            pybamm.laplacian(var) - pybamm.source(unit_source, var),
            pybamm.source(var, var),
            pybamm.laplacian(var) - pybamm.source(2 * var, var),
            pybamm.laplacian(var) - pybamm.source(unit_source ** 2 + 1 / var, var),
            pybamm.Integral(var, [y, z]) - 1,
        ]:
            # Check that equation can be evaluated in each case
            # Dirichlet
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # Neumann
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # One of each
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(1), "Dirichlet"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)
            # One of each
            disc.bcs = {
                var.id: {
                    "left": (pybamm.Scalar(0), "Dirichlet"),
                    "right": (pybamm.Scalar(1), "Neumann"),
                }
            }
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)

        # check  ValueError raised for non Dirichlet or Neumann BCs
        eqn = pybamm.laplacian(var) - pybamm.source(unit_source, var)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(1), "Other BC"),
            }
        }
        with self.assertRaises(ValueError):
            eqn_disc = disc.process_symbol(eqn)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(0), "Other BC"),
                "right": (pybamm.Scalar(1), "Neumann"),
            }
        }
        with self.assertRaises(ValueError):
            eqn_disc = disc.process_symbol(eqn)

    def test_manufactured_solution(self):
        mesh = get_unit_2p1D_mesh_for_testing(ypts=32, zpts=32)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.ScikitFiniteElement,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # linear u = z (to test coordinates to degree of freedom mapping)
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])
        var_disc = disc.process_symbol(var)
        z_vertices = mesh["current collector"][0].coordinates[1, :]
        np.testing.assert_array_almost_equal(
            var_disc.evaluate(None, z_vertices), z_vertices[:, np.newaxis]
        )

        # linear u = 6*y (to test coordinates to degree of freedom mapping)
        y_vertices = mesh["current collector"][0].coordinates[0, :]
        np.testing.assert_array_almost_equal(
            var_disc.evaluate(None, 6 * y_vertices), 6 * y_vertices[:, np.newaxis]
        )

        # mixed u = y*z (to test coordinates to degree of freedom mapping)
        np.testing.assert_array_almost_equal(
            var_disc.evaluate(None, y_vertices * z_vertices),
            y_vertices[:, np.newaxis] * z_vertices[:, np.newaxis],
        )

        # laplace of u = sin(pi*z)
        var = pybamm.Variable("var", domain="current collector")
        eqn_zz = pybamm.laplacian(var)
        # set boundary conditions ("left" = bottom of unit square, "right" = top
        # of unit square, elsewhere normal derivative is zero)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.set_variable_slices([var])
        eqn_zz_disc = disc.process_symbol(eqn_zz)
        z_vertices = mesh["current collector"][0].coordinates[1, :][:, np.newaxis]
        u = np.sin(np.pi * z_vertices)
        mass = pybamm.Mass(var)
        mass_disc = disc.process_symbol(mass)
        soln = -np.pi ** 2 * u
        np.testing.assert_array_almost_equal(
            eqn_zz_disc.evaluate(None, u), mass_disc.entries @ soln, decimal=3
        )

        # laplace of u = cos(pi*y)*sin(pi*z)
        var = pybamm.Variable("var", domain="current collector")
        laplace_eqn = pybamm.laplacian(var)
        # set boundary conditions ("left" = bottom of unit square, "right" = top
        # of unit square, elsewhere normal derivative is zero)
        disc.bcs = {
            var.id: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }
        disc.set_variable_slices([var])
        laplace_eqn_disc = disc.process_symbol(laplace_eqn)
        y_vertices = mesh["current collector"][0].coordinates[0, :][:, np.newaxis]
        z_vertices = mesh["current collector"][0].coordinates[1, :][:, np.newaxis]
        u = np.cos(np.pi * y_vertices) * np.sin(np.pi * z_vertices)
        mass = pybamm.Mass(var)
        mass_disc = disc.process_symbol(mass)
        soln = -np.pi ** 2 * u
        np.testing.assert_array_almost_equal(
            laplace_eqn_disc.evaluate(None, u), mass_disc.entries @ soln, decimal=2
        )

    def test_definite_integral(self):
        mesh = get_2p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.ScikitFiniteElement,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="current collector")
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        integral_eqn = pybamm.Integral(var, [y, z])
        disc.set_variable_slices([var])
        integral_eqn_disc = disc.process_symbol(integral_eqn)
        y_test = 6 * np.ones(mesh["current collector"][0].npts)
        fem_mesh = mesh["current collector"][0]
        ly = fem_mesh.coordinates[0, -1]
        lz = fem_mesh.coordinates[1, -1]
        np.testing.assert_array_almost_equal(
            integral_eqn_disc.evaluate(None, y_test), 6 * ly * lz
        )

    def test_definite_integral_vector(self):
        mesh = get_2p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.ScikitFiniteElement,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])

        # row (default)
        vec = pybamm.DefiniteIntegralVector(var)
        vec_disc = disc.process_symbol(vec)
        self.assertEqual(vec_disc.shape[0], 1)
        self.assertEqual(vec_disc.shape[1], mesh["current collector"][0].npts)

        # column
        vec = pybamm.DefiniteIntegralVector(var, vector_type="column")
        vec_disc = disc.process_symbol(vec)
        self.assertEqual(vec_disc.shape[0], mesh["current collector"][0].npts)
        self.assertEqual(vec_disc.shape[1], 1)

    def test_left_right(self):
        mesh = get_2p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.ScikitFiniteElement,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])

        extrap_left = pybamm.BoundaryValue(var, "left")
        extrap_right = pybamm.BoundaryValue(var, "right")
        extrap_left_disc = disc.process_symbol(extrap_left)
        extrap_right_disc = disc.process_symbol(extrap_right)

        # check constant returns constant at tab
        constant_y = np.ones(mesh["current collector"][0].npts)
        self.assertEqual(extrap_left_disc.evaluate(None, constant_y), 1)
        self.assertEqual(extrap_right_disc.evaluate(None, constant_y), 1)

    def test_boundary_integral(self):
        mesh = get_2p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.ScikitFiniteElement,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        var = pybamm.Variable("var", domain="current collector")
        disc.set_variable_slices([var])

        full = pybamm.BoundaryIntegral(var)
        neg = pybamm.BoundaryIntegral(var, region="negative tab")
        pos = pybamm.BoundaryIntegral(var, region="positive tab")

        full_disc = disc.process_symbol(full)
        neg_disc = disc.process_symbol(neg)
        pos_disc = disc.process_symbol(pos)

        # check integrating 1 gives correct *dimensionless* region lengths
        perimeter = 2 * (1 + 0.8)
        l_tab_n = 0.1 / 0.5
        l_tab_p = 0.1 / 0.5
        constant_y = np.ones(mesh["current collector"][0].npts)
        # Integral around boundary is exact
        np.testing.assert_array_almost_equal(
            full_disc.evaluate(None, constant_y), perimeter
        )
        # Ideally mesh edges should line up with tab edges.... then we would get
        # better agreement between actual and numerical tab width
        np.testing.assert_array_almost_equal(
            neg_disc.evaluate(None, constant_y), l_tab_n, decimal=1
        )
        np.testing.assert_array_almost_equal(
            pos_disc.evaluate(None, constant_y), l_tab_p, decimal=1
        )

    def test_pure_neumann_poisson(self):
        # grad^2 u = 1, du/dz = 1 at z = 1, du/dn = 0 elsewhere, u has zero average
        u = pybamm.Variable("u", domain="current collector")
        c = pybamm.Variable("c")  # lagrange multiplier
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])

        model = pybamm.BaseModel()
        # 0*c hack otherwise gives KeyError
        model.algebraic = {
            u: pybamm.laplacian(u)
            - pybamm.source(1, u)
            + c * pybamm.DefiniteIntegralVector(u, vector_type="column"),
            c: pybamm.Integral(u, [y, z]) + 0 * c,
        }
        model.initial_conditions = {u: pybamm.Scalar(0), c: pybamm.Scalar(0)}
        # set boundary conditions ("left" = bottom of unit square, "right" = top
        # of unit square, elsewhere normal derivative is zero)
        model.boundary_conditions = {
            u: {"left": (0, "Neumann"), "right": (1, "Neumann")}
        }
        model.variables = {"c": c, "u": u}
        # create discretisation
        mesh = get_unit_2p1D_mesh_for_testing(ypts=32, zpts=32)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.ScikitFiniteElement,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # solve model
        solver = pybamm.AlgebraicSolver()
        solution = solver.solve(model)

        z = mesh["current collector"][0].coordinates[1, :][:, np.newaxis]
        u_exact = z ** 2 / 2 - 1 / 6
        np.testing.assert_array_almost_equal(solution.y[:-1], u_exact, decimal=1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
