#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

double pv_continuous(double face_value, double coupon_rate, double yield_rate, double years) {
    const double C = face_value * coupon_rate;
    const double r = yield_rate;
    const double T = years;
    if (r == 0.0) {
        return C * T + face_value;
    }
    const double discount = std::exp(-r * T);
    return C * (1.0 - discount) / r + face_value * discount;
}

py::list discount_factors_continuous(double yield_rate, py::iterable tenors) {
    py::list out;
    for (auto item : tenors) {
        const double t = py::float_(item);
        const double df = std::exp(-yield_rate * t);
        out.append(py::make_tuple(t, df));
    }
    return out;
}

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native helpers for bondviz implemented with pybind11.";

    m.def(
        "pv_continuous",
        &pv_continuous,
        py::arg("face_value"),
        py::arg("coupon_rate"),
        py::arg("yield_rate"),
        py::arg("years"),
        "Present value of a fixed coupon bond under continuous compounding."
    );

    m.def(
        "discount_factors_continuous",
        &discount_factors_continuous,
        py::arg("yield_rate"),
        py::arg("tenors"),
        "Vector of (tenor, discount factor) pairs under continuous compounding."
    );
}

