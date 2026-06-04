#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

double pv_continuous(double face_value, double coupon_rate, double yield_rate,
                     double years, int freq) {
    // Discrete coupon schedule discounted under continuous compounding, kept
    // numerically identical to pricing._bond_cashflows + price_from_cashflows.
    int n = static_cast<int>(std::lround(years * freq));
    if (n < 1) {
        n = 1;
    }
    const double coupon_pmt = face_value * coupon_rate / freq;
    double pv = 0.0;
    for (int i = 1; i <= n; ++i) {
        const double t = static_cast<double>(i) / freq;
        double cf = coupon_pmt;
        if (i == n) {
            cf += face_value;
        }
        pv += cf * std::exp(-yield_rate * t);
    }
    return pv;
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
        py::arg("freq") = 2,
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

