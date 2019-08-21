#
# Tests for the base battery model class
#
import pybamm
import unittest


class TestBaseBatteryModel(unittest.TestCase):
    def test_process_parameters_and_discretise(self):
        model = pybamm.ReactionDiffusionModel()
        c = pybamm.Parameter("Negative electrode width [m]") * pybamm.Variable(
            "Negative electrolyte concentration",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        processed_c = model.process_parameters_and_discretise(c)
        self.assertIsInstance(processed_c, pybamm.Multiplication)
        self.assertIsInstance(processed_c.left, pybamm.Scalar)
        self.assertIsInstance(processed_c.right, pybamm.StateVector)

    def test_default_geometry(self):
        var = pybamm.standard_spatial_vars

        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        self.assertEqual(
            model.default_geometry["current collector"]["primary"][var.z][
                "position"
            ].id,
            pybamm.Scalar(1).id,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        self.assertEqual(
            model.default_geometry["current collector"]["primary"][var.z]["min"].id,
            pybamm.Scalar(0).id,
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        self.assertEqual(
            model.default_geometry["current collector"]["primary"][var.y]["min"].id,
            pybamm.Scalar(0).id,
        )

    def test_default_submesh_types(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"], pybamm.SubMesh0D
            )
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"],
                pybamm.Uniform1DSubMesh,
            )
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"], pybamm.Scikit2DSubMesh
            )
        )

    def test_default_spatial_methods(self):
        model = pybamm.BaseBatteryModel({"dimensionality": 0})
        self.assertTrue(
            issubclass(
                model.default_spatial_methods["current collector"],
                pybamm.ZeroDimensionalMethod,
            )
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 1})
        self.assertTrue(
            issubclass(
                model.default_spatial_methods["current collector"], pybamm.FiniteVolume
            )
        )
        model = pybamm.BaseBatteryModel({"dimensionality": 2})
        self.assertTrue(
            issubclass(
                model.default_spatial_methods["current collector"],
                pybamm.ScikitFiniteElement,
            )
        )

    def test_bad_options(self):
        with self.assertRaisesRegex(pybamm.OptionError, "current collector model"):
            pybamm.BaseBatteryModel({"current collector": "bad current collector"})
        with self.assertRaisesRegex(pybamm.OptionError, "thermal model"):
            pybamm.BaseBatteryModel({"thermal": "bad thermal"})
        with self.assertRaisesRegex(
            pybamm.OptionError, "Dimension of current collectors"
        ):
            pybamm.BaseBatteryModel({"dimensionality": 5})
        with self.assertRaisesRegex(pybamm.OptionError, "surface form"):
            pybamm.BaseBatteryModel({"surface form": "bad surface form"})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
