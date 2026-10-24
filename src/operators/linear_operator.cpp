#include "declib/operators/linear_operator.hpp"

#include <utility>

#include "declib/core/error.hpp"
#include "declib/forms/form_type.hpp"
#include "declib/operators/pairing.hpp"
#include "detail/petsc.hpp"

// ---------------------------------------------------------------- shell
// PETSc's error macros call PetscError unqualified; the callbacks live
// outside namespace declib so that nothing there can shadow it.

namespace {


struct ShellContext {
    declib::OpPtr op;
};

PetscErrorCode shell_mult(Mat A, Vec x, Vec y)
{
    ShellContext* ctx;
    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(A, &ctx));
    try {
        const declib::Form fx = declib::Form::view_of(ctx->op->domain(), x);
        declib::Form fy = declib::Form::view_of(ctx->op->range(), y);
        ctx->op->apply(fx, fy);
    } catch (const std::exception& e) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", e.what());
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode shell_mult_transpose(Mat A, Vec y, Vec x)
{
    ShellContext* ctx;
    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(A, &ctx));
    try {
        const declib::Form fy = declib::Form::view_of(ctx->op->range(), y);
        declib::Form fx = declib::Form::view_of(ctx->op->domain(), x);
        ctx->op->apply_transpose(fy, fx);
    } catch (const std::exception& e) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", e.what());
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode shell_destroy(void* ctx)
{
    PetscFunctionBeginUser;
    delete static_cast<ShellContext*>(ctx);
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

namespace declib {

namespace {

// RAII for matrices produced inside composites.
struct MatHolder {
    Mat m = nullptr;
    MatHolder() = default;
    explicit MatHolder(Mat a) : m(a) {}
    MatHolder(const MatHolder&) = delete;
    MatHolder& operator=(const MatHolder&) = delete;
    ~MatHolder() { MatDestroy(&m); }
    Mat release() { return std::exchange(m, nullptr); }
};

struct VecHolder {
    Vec v = nullptr;
    VecHolder() = default;
    VecHolder(const VecHolder&) = delete;
    VecHolder& operator=(const VecHolder&) = delete;
    ~VecHolder() { VecDestroy(&v); }
};

Mat convert(Mat A, MatType type)
{
    PetscBool same;
    DECLIB_PETSC(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), type, &same));
    if (same) return A;
    Mat B;
    DECLIB_PETSC(MatConvert(A, type, MAT_INITIAL_MATRIX, &B));
    DECLIB_PETSC(MatDestroy(&A));
    return B;
}

// Diagonal of an operator known to be diagonal, as a vector in its range
// layout whose type matches matrices of type `type`.
Vec diagonal_of(const LinearOperator& A, MatType type = MATAIJKOKKOS)
{
    MatHolder m(A.assemble(type));
    Vec d;
    DECLIB_PETSC(MatCreateVecs(m.m, nullptr, &d));
    DECLIB_PETSC(MatGetDiagonal(m.m, d));
    return d;
}

void require_same(const FormSpace& a, const FormSpace& b, const char* what)
{
    if (!a.same_type_as(b))
        throw TypeError(std::string(what) + ": " + to_string(a.type()) + " does not match " + to_string(b.type()));
}

// ---------------------------------------------------------------- composites

class Compose final : public LinearOperator {
public:
    Compose(OpPtr A, OpPtr B) : A_(std::move(A)), B_(std::move(B))
    {
        require_same(*A_->domain(), *B_->range(), "operator composition");
    }
    const SpacePtr& domain() const noexcept override { return B_->domain(); }
    const SpacePtr& range() const noexcept override { return A_->range(); }
    void apply(const Form& x, Form& y) const override
    {
        Form t = B_->range()->zeros();
        B_->apply(x, t);
        A_->apply(t, y);
    }
    void apply_transpose(const Form& y, Form& x) const override
    {
        Form t = B_->range()->zeros();
        A_->apply_transpose(y, t);
        B_->apply_transpose(t, x);
    }
    OpPtr pairing_adjoint() const override { return B_->pairing_adjoint() * A_->pairing_adjoint(); }
    Mat assemble(MatType type) const override
    {
        if (B_->is_diagonal()) {
            MatHolder a(A_->assemble(type));
            VecHolder d;
            d.v = diagonal_of(*B_, type);
            DECLIB_PETSC(MatDiagonalScale(a.m, nullptr, d.v));
            return a.release();
        }
        if (A_->is_diagonal()) {
            MatHolder b(B_->assemble(type));
            VecHolder d;
            d.v = diagonal_of(*A_, type);
            DECLIB_PETSC(MatDiagonalScale(b.m, d.v, nullptr));
            return b.release();
        }
        MatHolder a(A_->assemble(type)), b(B_->assemble(type));
        Mat c;
        DECLIB_PETSC(MatMatMult(a.m, b.m, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &c));
        return convert(c, type);
    }
    bool is_diagonal() const noexcept override { return A_->is_diagonal() && B_->is_diagonal(); }
    std::string name() const override { return "(" + A_->name() + " * " + B_->name() + ")"; }

private:
    OpPtr A_, B_;
};

// a A + b B
class Combination final : public LinearOperator {
public:
    Combination(Scalar a, OpPtr A, Scalar b, OpPtr B) : a_(a), b_(b), A_(std::move(A)), B_(std::move(B))
    {
        require_same(*A_->domain(), *B_->domain(), "operator sum (domains)");
        require_same(*A_->range(), *B_->range(), "operator sum (ranges)");
    }
    const SpacePtr& domain() const noexcept override { return A_->domain(); }
    const SpacePtr& range() const noexcept override { return A_->range(); }
    void apply(const Form& x, Form& y) const override
    {
        A_->apply(x, y);
        if (a_ != Scalar(1)) y.scale(a_);
        B_->apply_add(x, y, b_);
    }
    void apply_transpose(const Form& y, Form& x) const override
    {
        A_->apply_transpose(y, x);
        if (a_ != Scalar(1)) x.scale(a_);
        Form t = A_->domain()->zeros();
        B_->apply_transpose(y, t);
        x.axpy(b_, t);
    }
    OpPtr pairing_adjoint() const override
    {
        return std::make_shared<const Combination>(a_, A_->pairing_adjoint(), b_, B_->pairing_adjoint());
    }
    Mat assemble(MatType type) const override
    {
        MatHolder a(A_->assemble(type)), b(B_->assemble(type));
        DECLIB_PETSC(MatScale(a.m, a_));
        DECLIB_PETSC(MatAXPY(a.m, b_, b.m, DIFFERENT_NONZERO_PATTERN));
        return a.release();
    }
    bool is_diagonal() const noexcept override { return A_->is_diagonal() && B_->is_diagonal(); }
    std::string name() const override
    {
        return "(" + std::to_string(a_) + " " + A_->name() + " + " + std::to_string(b_) + " " + B_->name() + ")";
    }

private:
    Scalar a_, b_;
    OpPtr A_, B_;
};

class Scaled final : public LinearOperator {
public:
    Scaled(Scalar a, OpPtr A) : a_(a), A_(std::move(A)) {}
    const SpacePtr& domain() const noexcept override { return A_->domain(); }
    const SpacePtr& range() const noexcept override { return A_->range(); }
    void apply(const Form& x, Form& y) const override
    {
        A_->apply(x, y);
        y.scale(a_);
    }
    void apply_transpose(const Form& y, Form& x) const override
    {
        A_->apply_transpose(y, x);
        x.scale(a_);
    }
    OpPtr pairing_adjoint() const override { return a_ * A_->pairing_adjoint(); }
    Mat assemble(MatType type) const override
    {
        MatHolder a(A_->assemble(type));
        DECLIB_PETSC(MatScale(a.m, a_));
        return a.release();
    }
    bool is_diagonal() const noexcept override { return A_->is_diagonal(); }
    std::string name() const override { return std::to_string(a_) + " " + A_->name(); }

private:
    Scalar a_;
    OpPtr A_;
};

class Identity final : public LinearOperator {
public:
    explicit Identity(SpacePtr space) : space_(std::move(space)) {}
    const SpacePtr& domain() const noexcept override { return space_; }
    const SpacePtr& range() const noexcept override { return space_; }
    void apply(const Form& x, Form& y) const override { DECLIB_PETSC(VecCopy(x.global_vec(), y.global_vec())); }
    void apply_transpose(const Form& y, Form& x) const override { apply(y, x); }
    OpPtr pairing_adjoint() const override { return identity(space_->dual_space()); }
    Mat assemble(MatType type) const override
    {
        const Form f = space_->zeros();
        PetscInt lo, hi;
        DECLIB_PETSC(VecGetOwnershipRange(f.global_vec(), &lo, &hi));
        std::vector<PetscInt> idx(static_cast<std::size_t>(hi - lo));
        for (PetscInt i = lo; i < hi; ++i) idx[static_cast<std::size_t>(i - lo)] = i;
        std::vector<PetscScalar> ones(idx.size(), 1);
        Mat A;
        DECLIB_PETSC(MatCreate(PetscObjectComm(reinterpret_cast<PetscObject>(f.global_vec())), &A));
        DECLIB_PETSC(MatSetSizes(A, hi - lo, hi - lo, PETSC_DETERMINE, PETSC_DETERMINE));
        DECLIB_PETSC(MatSetType(A, type));
        DECLIB_PETSC(MatSetPreallocationCOO(A, static_cast<PetscCount>(idx.size()), idx.data(), idx.data()));
        DECLIB_PETSC(MatSetValuesCOO(A, ones.data(), INSERT_VALUES));
        return A;
    }
    bool is_diagonal() const noexcept override { return true; }
    std::string name() const override { return "I"; }

private:
    SpacePtr space_;
};

class Transposed final : public LinearOperator {
public:
    explicit Transposed(OpPtr A) : A_(std::move(A)) {}
    const SpacePtr& domain() const noexcept override { return A_->range(); }
    const SpacePtr& range() const noexcept override { return A_->domain(); }
    void apply(const Form& x, Form& y) const override { A_->apply_transpose(x, y); }
    void apply_transpose(const Form& y, Form& x) const override { A_->apply(y, x); }
    Mat assemble(MatType type) const override
    {
        MatHolder a(A_->assemble(type));
        Mat t;
        DECLIB_PETSC(MatTranspose(a.m, MAT_INITIAL_MATRIX, &t));
        return convert(t, type);
    }
    bool is_diagonal() const noexcept override { return A_->is_diagonal(); }
    std::string name() const override { return A_->name() + "^T"; }

private:
    OpPtr A_;
};

// A^dagger = S_V A^T S_W, with S the sign pattern of each pairing.
class PairingAdjoint final : public LinearOperator {
public:
    explicit PairingAdjoint(OpPtr A)
        : A_(std::move(A)), domain_(A_->range()->dual_space()), range_(A_->domain()->dual_space())
    {
        sv_.v = Pairing::signs(*A_->domain());
        sw_.v = Pairing::signs(*A_->range());
    }
    const SpacePtr& domain() const noexcept override { return domain_; }
    const SpacePtr& range() const noexcept override { return range_; }
    void apply(const Form& x, Form& y) const override
    {
        Form t = A_->range()->zeros();
        DECLIB_PETSC(VecPointwiseMult(t.global_vec(), sw_.v, x.global_vec()));
        Form u = A_->domain()->zeros();
        A_->apply_transpose(t, u);
        DECLIB_PETSC(VecPointwiseMult(y.global_vec(), sv_.v, u.global_vec()));
    }
    void apply_transpose(const Form& y, Form& x) const override
    {
        Form t = A_->domain()->zeros();
        DECLIB_PETSC(VecPointwiseMult(t.global_vec(), sv_.v, y.global_vec()));
        Form u = A_->range()->zeros();
        A_->apply(t, u);
        DECLIB_PETSC(VecPointwiseMult(x.global_vec(), sw_.v, u.global_vec()));
    }
    OpPtr pairing_adjoint() const override { return A_; }
    Mat assemble(MatType type) const override
    {
        MatHolder a(A_->assemble(type));
        Mat t;
        DECLIB_PETSC(MatTranspose(a.m, MAT_INITIAL_MATRIX, &t));
        DECLIB_PETSC(MatDiagonalScale(t, sv_.v, sw_.v));
        return convert(t, type);
    }
    bool is_diagonal() const noexcept override { return A_->is_diagonal(); }
    std::string name() const override { return A_->name() + "^dagger"; }

private:
    OpPtr A_;
    SpacePtr domain_, range_;
    VecHolder sv_, sw_;
};

class Inverse final : public LinearOperator {
public:
    Inverse(OpPtr A, KSPOptions options) : A_(std::move(A)), options_(std::move(options))
    {
        if (A_->domain()->global_size() != A_->range()->global_size())
            throw TypeError("inverse: operator " + A_->name() + " is not square");
        if (A_->is_diagonal()) {
            diag_.v = diagonal_of(*A_);
            DECLIB_PETSC(VecReciprocal(diag_.v));
            return;
        }
        DECLIB_PETSC(KSPCreate(PetscObjectComm(reinterpret_cast<PetscObject>(A_->domain()->dm())), &ksp_));
        if (options_.preconditioner) {
            // Matrix-free operator, assembled preconditioner.
            amat_.m = A_->shell();
            pmat_.m = options_.preconditioner->assemble(MATAIJKOKKOS);
            DECLIB_PETSC(KSPSetOperators(ksp_, amat_.m, pmat_.m));
        } else {
            amat_.m = A_->assemble(MATAIJKOKKOS);
            DECLIB_PETSC(KSPSetOperators(ksp_, amat_.m, amat_.m));
        }
        if (!options_.prefix.empty()) DECLIB_PETSC(KSPSetOptionsPrefix(ksp_, options_.prefix.c_str()));
        DECLIB_PETSC(KSPSetTolerances(ksp_, options_.rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
        DECLIB_PETSC(KSPSetFromOptions(ksp_));
    }
    ~Inverse() override { KSPDestroy(&ksp_); }
    const SpacePtr& domain() const noexcept override { return A_->range(); }
    const SpacePtr& range() const noexcept override { return A_->domain(); }
    void apply(const Form& x, Form& y) const override
    {
        if (diag_.v) {
            DECLIB_PETSC(VecPointwiseMult(y.global_vec(), diag_.v, x.global_vec()));
            return;
        }
        DECLIB_PETSC(KSPSolve(ksp_, x.global_vec(), y.global_vec()));
        check();
    }
    void apply_transpose(const Form& y, Form& x) const override
    {
        if (diag_.v) {
            apply(y, x);
            return;
        }
        DECLIB_PETSC(KSPSolveTranspose(ksp_, y.global_vec(), x.global_vec()));
        check();
    }
    OpPtr pairing_adjoint() const override { return inverse(A_->pairing_adjoint(), options_); }
    Mat assemble(MatType type) const override
    {
        if (!diag_.v) throw Error("inverse(" + A_->name() + ") is dense and cannot be assembled");
        MatHolder a(A_->assemble(type));
        DECLIB_PETSC(MatZeroEntries(a.m));
        DECLIB_PETSC(MatDiagonalSet(a.m, diag_.v, INSERT_VALUES));
        return a.release();
    }
    bool is_diagonal() const noexcept override { return A_->is_diagonal(); }
    std::string name() const override { return "inverse(" + A_->name() + ")"; }

private:
    void check() const
    {
        KSPConvergedReason reason;
        DECLIB_PETSC(KSPGetConvergedReason(ksp_, &reason));
        if (reason < 0)
            throw Error("inverse(" + A_->name() + "): KSP diverged (" + KSPConvergedReasons[reason] + ")");
    }

    OpPtr A_;
    KSPOptions options_;
    VecHolder diag_;
    MatHolder amat_, pmat_;
    KSP ksp_ = nullptr;
};

} // namespace

// ---------------------------------------------------------------- base class

LinearOperator::~LinearOperator() { MatDestroy(&cached_); }

void LinearOperator::apply_add(const Form& x, Form& y, Scalar a) const
{
    Form t = range()->zeros();
    apply(x, t);
    y.axpy(a, t);
}

void LinearOperator::apply_transpose(const Form& y, Form& x) const
{
    DECLIB_PETSC(MatMultTranspose(cached_matrix(), y.global_vec(), x.global_vec()));
}

Form LinearOperator::operator()(const Form& x) const
{
    Form y = range()->zeros();
    check_args(x, y);
    apply(x, y);
    return y;
}

OpPtr LinearOperator::pairing_adjoint() const { return std::make_shared<const PairingAdjoint>(shared_from_this()); }

Mat LinearOperator::shell() const
{
    const Form x = domain()->zeros();
    const Form y = range()->zeros();
    PetscInt m, n, M, N;
    DECLIB_PETSC(VecGetLocalSize(y.global_vec(), &m));
    DECLIB_PETSC(VecGetLocalSize(x.global_vec(), &n));
    DECLIB_PETSC(VecGetSize(y.global_vec(), &M));
    DECLIB_PETSC(VecGetSize(x.global_vec(), &N));
    auto* ctx = new ShellContext{shared_from_this()};
    Mat A;
    PetscErrorCode ierr = MatCreateShell(PetscObjectComm(reinterpret_cast<PetscObject>(x.global_vec())), m, n, M, N,
                                         ctx, &A);
    if (ierr) {
        delete ctx;
        detail::throw_petsc(ierr, "MatCreateShell", __FILE__, __LINE__);
    }
    DECLIB_PETSC(MatShellSetContextDestroy(A, shell_destroy));
    DECLIB_PETSC(MatShellSetOperation(A, MATOP_MULT, reinterpret_cast<void (*)(void)>(shell_mult)));
    DECLIB_PETSC(
        MatShellSetOperation(A, MATOP_MULT_TRANSPOSE, reinterpret_cast<void (*)(void)>(shell_mult_transpose)));
    DECLIB_PETSC(MatShellSetVecType(A, VECKOKKOS));
    return A;
}

Mat LinearOperator::assemble(MatType type) const
{
    // Generic fallback: probe the shell column by column.
    MatHolder s(shell());
    Mat A;
    DECLIB_PETSC(MatComputeOperator(s.m, MATAIJ, &A));
    return convert(A, type);
}

std::string LinearOperator::name() const { return "LinearOperator"; }

Mat LinearOperator::cached_matrix() const
{
    if (!cached_) cached_ = assemble(MATAIJKOKKOS);
    return cached_;
}

void LinearOperator::check_args(const Form& x, const Form& y) const
{
    if (!x.space().same_type_as(*domain()))
        throw TypeError(name() + ": input is a " + to_string(x.type()) + ", expected a " +
                        to_string(domain()->type()));
    if (!y.space().same_type_as(*range()))
        throw TypeError(name() + ": output is a " + to_string(y.type()) + ", expected a " +
                        to_string(range()->type()));
}

// ---------------------------------------------------------------- algebra

OpPtr operator*(OpPtr A, OpPtr B)
{
    if (!A || !B) throw Error("operator*: null operator");
    return std::make_shared<const Compose>(std::move(A), std::move(B));
}

OpPtr operator+(OpPtr A, OpPtr B)
{
    if (!A || !B) throw Error("operator+: null operator");
    return std::make_shared<const Combination>(Scalar(1), std::move(A), Scalar(1), std::move(B));
}

OpPtr operator-(OpPtr A, OpPtr B)
{
    if (!A || !B) throw Error("operator-: null operator");
    return std::make_shared<const Combination>(Scalar(1), std::move(A), Scalar(-1), std::move(B));
}

OpPtr operator*(Scalar a, OpPtr A)
{
    if (!A) throw Error("operator*: null operator");
    return std::make_shared<const Scaled>(a, std::move(A));
}

OpPtr inverse(OpPtr A, KSPOptions options)
{
    if (!A) throw Error("inverse: null operator");
    return std::make_shared<const Inverse>(std::move(A), std::move(options));
}

OpPtr identity(SpacePtr space)
{
    if (!space) throw Error("identity: null space");
    return std::make_shared<const Identity>(std::move(space));
}

OpPtr transpose(OpPtr A)
{
    if (!A) throw Error("transpose: null operator");
    return std::make_shared<const Transposed>(std::move(A));
}

} // namespace declib
