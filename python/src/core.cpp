// pybind11 bindings: the private extension declib._core. The public Python
// API lives in the pure-Python package declib, which wraps these classes.
// PETSc objects cross the boundary as petsc4py objects.

#include <petsc4py/petsc4py.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "declib/declib.hpp"

namespace py = pybind11;
using namespace declib;

namespace {

// ---------------------------------------------------------------- petsc4py

py::object wrap(Vec v) { return py::reinterpret_steal<py::object>(PyPetscVec_New(v)); }
py::object wrap(DM dm) { return py::reinterpret_steal<py::object>(PyPetscDM_New(dm)); }

// Takes over the caller's reference.
py::object wrap_owned(Mat m)
{
    py::object o = py::reinterpret_steal<py::object>(PyPetscMat_New(m));
    MatDestroy(&m);
    return o;
}

py::object wrap_owned(IS is)
{
    py::object o = py::reinterpret_steal<py::object>(PyPetscIS_New(is));
    ISDestroy(&is);
    return o;
}

Mat as_mat(const py::object& o)
{
    Mat m = PyPetscMat_Get(o.ptr());
    if (!m) throw py::error_already_set();
    return m;
}

MPI_Comm as_comm(const py::object& comm)
{
    if (comm.is_none()) return PETSC_COMM_WORLD;
    // mpi4py communicators, and petsc4py ones through tompi4py().
    py::object c = py::hasattr(comm, "tompi4py") ? comm.attr("tompi4py")() : comm;
    return MPI_Comm_f2c(c.attr("py2f")().cast<MPI_Fint>());
}

// ---------------------------------------------------------------- enums

Orientation orientation(const std::string& s)
{
    if (s == "straight") return Orientation::Straight;
    if (s == "twisted") return Orientation::Twisted;
    throw py::value_error("unknown orientation " + s);
}

Complex complex_of(const std::string& s)
{
    if (s == "primal") return Complex::Primal;
    if (s == "dual") return Complex::Dual;
    throw py::value_error("unknown complex " + s);
}

HodgeScheme scheme(const std::string& s)
{
    if (s == "diagonal") return HodgeScheme::Diagonal;
    if (s == "galerkin") return HodgeScheme::Galerkin;
    if (s == "high_order") return HodgeScheme::HighOrder;
    throw py::value_error("unknown Hodge scheme " + s);
}

stag::Boundary boundary(const std::string& s)
{
    if (s == "periodic") return stag::Boundary::Periodic;
    if (s == "ghosted") return stag::Boundary::Ghosted;
    if (s == "none") return stag::Boundary::None;
    throw py::value_error("unknown boundary type " + s);
}

Bundle bundle(const std::string& kind, int rank, const std::string& fiber)
{
    if (kind == "scalar") return Bundle::scalar();
    if (kind == "trivial") return Bundle::trivial(rank);
    if (kind == "tangent") return Bundle::tangent(rank);
    if (kind == "cotangent") return Bundle::cotangent(rank, orientation(fiber));
    throw py::value_error("unsupported bundle kind " + kind);
}

CellSet cellset(const std::string& c, int k) { return CellSet{complex_of(c), k}; }

// ---------------------------------------------------------------- handles

struct PyTopology {
    std::shared_ptr<const Topology> p;
};

struct PyGeometry {
    std::shared_ptr<const Geometry> p;
};

struct PyManifold {
    std::shared_ptr<const DiscreteManifold> p;
};

struct PySpace {
    SpacePtr p;
};

struct PyForm {
    std::shared_ptr<Form> p;
};

struct PyOp {
    OpPtr p;
};

PyTopology make_stag_topology(const std::vector<PetscInt>& cells, const std::vector<std::string>& bnd,
                              const std::string& dual_boundary, int stencil_width, const py::object& comm)
{
    const auto build = [&]<int Dim>() -> PyTopology {
        typename stag::Topology<Dim>::Options o;
        for (int a = 0; a < Dim; ++a) {
            o.cells[a] = cells[a];
            o.boundary[a] = boundary(bnd[a]);
        }
        o.dual_boundary = dual_boundary == "truncated" ? DualBoundary::Truncated : DualBoundary::Extended;
        o.stencil_width = stencil_width;
        return {stag::Topology<Dim>::create(as_comm(comm), o)};
    };
    if (cells.size() != bnd.size()) throw py::value_error("need one boundary type per axis");
    switch (cells.size()) {
    case 1: return build.template operator()<1>();
    case 2: return build.template operator()<2>();
    case 3: return build.template operator()<3>();
    default: throw py::value_error("StagTopology supports 1 to 3 dimensions");
    }
}

PyGeometry make_uniform_geometry(const PyTopology& t, const std::vector<Real>& lengths, const std::vector<Real>& origin)
{
    const auto build = [&]<int Dim>() -> PyGeometry {
        auto topo = std::dynamic_pointer_cast<const stag::Topology<Dim>>(t.p);
        if (!topo) throw TypeError("UniformGeometry needs a structured topology");
        typename stag::UniformGeometry<Dim>::Options o;
        for (int a = 0; a < Dim; ++a) {
            o.lengths[a] = lengths.at(a);
            o.origin[a] = origin.at(a);
        }
        return {stag::UniformGeometry<Dim>::create(topo, o)};
    };
    switch (t.p->dimension()) {
    case 1: return build.template operator()<1>();
    case 2: return build.template operator()<2>();
    default: return build.template operator()<3>();
    }
}

Real spacing(const PyGeometry& g, int axis)
{
    if (auto* u = dynamic_cast<const stag::UniformGeometry<1>*>(g.p.get())) return u->spacing(axis);
    if (auto* u = dynamic_cast<const stag::UniformGeometry<2>*>(g.p.get())) return u->spacing(axis);
    if (auto* u = dynamic_cast<const stag::UniformGeometry<3>*>(g.p.get())) return u->spacing(axis);
    throw TypeError("spacing() needs a UniformGeometry");
}

// omega(*x) returns a float or a sequence of components.
FormFunction form_function(py::object omega)
{
    return [omega](std::span<const Real> x, std::span<Scalar> out) {
        py::gil_scoped_acquire gil;
        py::tuple args(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) args[i] = py::float_(x[i]);
        py::object r = omega(*args);
        if (py::isinstance<py::float_>(r) || py::isinstance<py::int_>(r)) {
            if (out.size() != 1) throw TypeError("form function returned one value, expected " +
                                                 std::to_string(out.size()));
            out[0] = r.cast<Scalar>();
            return;
        }
        const auto v = r.cast<std::vector<Scalar>>();
        if (v.size() != out.size())
            throw TypeError("form function returned " + std::to_string(v.size()) + " values, expected " +
                            std::to_string(out.size()));
        std::copy(v.begin(), v.end(), out.begin());
    };
}

std::vector<EssentialCondition> conditions(const py::list& items)
{
    std::vector<EssentialCondition> out;
    for (const auto& item : items) {
        auto t = item.cast<py::tuple>();
        EssentialCondition c;
        c.region = t[0].cast<BoundaryRegion>();
        if (!t[1].is_none()) c.value = form_function(py::reinterpret_borrow<py::object>(t[1]));
        out.push_back(std::move(c));
    }
    return out;
}

} // namespace

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Compiled core of DecLib++ (private; use the declib package).";
    if (import_petsc4py() < 0) throw py::error_already_set();

    // Translators run most recent first, so the base class goes first.
    py::register_exception<declib::Error>(m, "Error", PyExc_RuntimeError);
    py::register_exception<declib::PetscException>(m, "PetscError", PyExc_RuntimeError);
    py::register_exception<declib::TypeError>(m, "TypeError", PyExc_TypeError);

    py::class_<PyTopology>(m, "Topology")
        .def("dimension", [](const PyTopology& t) { return t.p->dimension(); })
        .def("dual_boundary", [](const PyTopology& t) {
            return t.p->dual_boundary() == DualBoundary::Extended ? "extended" : "truncated";
        })
        .def("num_owned", [](const PyTopology& t, const std::string& c, int k) { return t.p->num_owned(cellset(c, k)); })
        .def("num_global", [](const PyTopology& t, const std::string& c, int k) { return t.p->num_global(cellset(c, k)); })
        .def("incidence", [](const PyTopology& t, const std::string& c, int k) {
            return wrap_owned(t.p->incidence(cellset(c, k)));
        });
    m.def("StagTopology", &make_stag_topology, py::arg("cells"), py::arg("boundary"), py::arg("dual_boundary"),
          py::arg("stencil_width"), py::arg("comm") = py::none());

    py::class_<PyGeometry>(m, "Geometry")
        .def("dual_placement", [](const PyGeometry& g) {
            return g.p->dual_placement() == DualPlacement::Circumcentric ? "circumcentric" : "barycentric";
        })
        .def("is_orthogonal", [](const PyGeometry& g) { return g.p->is_orthogonal(); })
        .def("spacing", &spacing);
    m.def("UniformGeometry", &make_uniform_geometry);

    py::class_<BoundaryRegion>(m, "BoundaryRegion")
        .def_static("all", &BoundaryRegion::all)
        .def_static("face", [](int axis, bool high) { return BoundaryRegion::face(axis, high ? Side::High : Side::Low); });

    py::class_<PyManifold>(m, "DiscreteManifold")
        .def(py::init([](const PyTopology& t, const PyGeometry& g, bool straight_on_primal) {
            return PyManifold{DiscreteManifold::create(t.p, g.p,
                                                       straight_on_primal ? GridMapping::straight_on_primal()
                                                                          : GridMapping::twisted_on_primal())};
        }))
        .def("space", [](const PyManifold& M, int k, const std::string& o, const std::string& kind, int rank,
                         const std::string& fiber) {
            return PySpace{M.p->space(k, orientation(o), bundle(kind, rank, fiber))};
        });

    py::class_<PySpace>(m, "FormSpace")
        .def("dm", [](const PySpace& s) { return wrap(s.p->dm()); })
        .def("global_size", [](const PySpace& s) { return s.p->global_size(); })
        .def("local_size", [](const PySpace& s) { return s.p->local_size(); })
        .def("complex", [](const PySpace& s) { return s.p->cells().complex == Complex::Primal ? "primal" : "dual"; })
        .def("create_form", [](const PySpace& s) { return PyForm{std::make_shared<Form>(s.p)}; })
        .def("de_rham", [](const PySpace& s, py::object omega, PyForm& f, int order) {
            s.p->manifold().geometry().de_rham(form_function(std::move(omega)), *f.p, Quadrature{order});
        })
        .def("with_essential", [](const PySpace& s, const py::list& c) { return PySpace{s.p->with_essential(conditions(c))}; })
        .def("num_essential", [](const PySpace& s) { return s.p->essential_conditions().size(); })
        .def("boundary_indices", [](const PySpace& s, const BoundaryRegion& r) { return wrap_owned(s.p->boundary_indices(r)); });

    py::class_<PyForm>(m, "Form")
        .def("global_vec", [](const PyForm& f) { return wrap(f.p->global_vec()); })
        .def("axpy", [](PyForm& f, Scalar a, const PyForm& x) { f.p->axpy(a, *x.p); })
        .def("scale", [](PyForm& f, Scalar a) { f.p->scale(a); })
        .def("fill", [](PyForm& f, Scalar a) { f.p->fill(a); })
        .def("copy_from", [](PyForm& f, const PyForm& x) {
            if (!x.p->space().same_type_as(f.p->space())) throw TypeError("copy between forms of different types");
            VecCopy(x.p->global_vec(), f.p->global_vec());
        })
        .def("norm2", [](const PyForm& f) { return f.p->norm2(); })
        .def("norm_max", [](const PyForm& f) { return f.p->norm_max(); });

    py::class_<PyOp>(m, "LinearOperator")
        .def("apply", [](const PyOp& A, const PyForm& x, PyForm& y) { A.p->apply(*x.p, *y.p); })
        .def("apply_transpose", [](const PyOp& A, const PyForm& y, PyForm& x) { A.p->apply_transpose(*y.p, *x.p); })
        .def("pairing_adjoint", [](const PyOp& A) { return PyOp{A.p->pairing_adjoint()}; })
        .def("shell", [](const PyOp& A) { return wrap_owned(A.p->shell()); })
        .def("assemble", [](const PyOp& A, const std::string& type) { return wrap_owned(A.p->assemble(type.c_str())); },
             py::arg("type") = std::string(MATAIJKOKKOS))
        .def("is_diagonal", [](const PyOp& A) { return A.p->is_diagonal(); })
        .def("name", [](const PyOp& A) { return A.p->name(); });

    m.def("compose", [](const PyOp& A, const PyOp& B) { return PyOp{A.p * B.p}; });
    m.def("add", [](const PyOp& A, const PyOp& B) { return PyOp{A.p + B.p}; });
    m.def("subtract", [](const PyOp& A, const PyOp& B) { return PyOp{A.p - B.p}; });
    m.def("scale", [](Scalar a, const PyOp& A) { return PyOp{a * A.p}; });
    m.def("transpose", [](const PyOp& A) { return PyOp{transpose(A.p)}; });
    m.def("identity", [](const PySpace& s) { return PyOp{identity(s.p)}; });
    m.def("inverse", [](const PyOp& A, Real rtol, const std::string& prefix) {
        return PyOp{inverse(A.p, KSPOptions{prefix, rtol, nullptr})};
    });
    m.def("ExteriorDerivative", [](const PySpace& s) { return PyOp{ExteriorDerivative::create(s.p)}; });
    m.def("HodgeStar", [](const PySpace& s, const std::string& sch) { return PyOp{HodgeStar::create(s.p, scheme(sch))}; });
    m.def("codifferential", [](const PySpace& s, const std::string& sch) { return PyOp{codifferential(s.p, scheme(sch))}; });
    m.def("hodge_laplacian", [](const PySpace& s, const std::string& sch) { return PyOp{hodge_laplacian(s.p, scheme(sch))}; });
    m.def("mass", [](const PySpace& s, const std::string& sch) { return PyOp{mass(s.p, scheme(sch))}; });
    m.def("apply_essential", [](const PySpace& s, const py::object& A, PyForm& x, PyForm& b) {
        apply_essential(*s.p, as_mat(A), *x.p, *b.p);
    });

    py::class_<Pairing, std::shared_ptr<Pairing>>(m, "Pairing")
        .def(py::init([](const PySpace& s) { return std::const_pointer_cast<Pairing>(Pairing::create(s.p)); }))
        .def("pair", [](const Pairing& P, const PyForm& a, const PyForm& b) { return P(*a.p, *b.p); });
}
