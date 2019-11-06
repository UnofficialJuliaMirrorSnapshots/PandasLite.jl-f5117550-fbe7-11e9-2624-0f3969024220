__precompile__(true)
module PandasLite

using PyCall
using Compat
using TableTraits
using Dates

export values, DataFrame, Series, np, pd
export @pytype, @pyasvec

const np = PyNULL()
const pd = PyNULL()

function __init__()
    copy!(np, pyimport_conda("numpy", "numpy"))
    copy!(pd, pyimport_conda("pandas", "pandas"))
    empty!(type_map)  # for behaving nicely in system image
    for (pandas_expr, julia_type) in pre_type_map
        type_map[pandas_expr()] = julia_type
    end
    for (pytype, jltype) in type_map
        PyCall.pytype_mapping(pytype, jltype)
    end
    PyCall.pytype_mapping(np.float32, Float32)
    PyCall.pytype_mapping(np.float64, Float64)
    PyCall.pytype_mapping(np.int32, Int32)
    PyCall.pytype_mapping(np.int64, Int64)
    if get(ENV, "PD_CONSOLID", "false") == "false"
        noconsolidation()
    end
end

const pre_type_map = []

# Maps a python object corresponding to a PandasLite class to a Julia type which
# wraps that class.
const type_map = Dict{PyObject, Type}()

abstract type PandasWrapped end

PyCall.PyObject(x::PandasWrapped) = x.pyo

macro pytype(name, class)
    quote
        mutable struct $(name) <: PandasWrapped
            pyo::PyObject
            $(esc(name))(pyo::PyObject) = new(pyo)
            $(esc(name))(args...; kwargs...) = ($class)()(args...; kwargs...)
        end

        # This won't work until PyCall is updated to support
        # the Julia 1.0 iteration protocol.
        function Base.iterate(x::$name, state...)
            res = Base.iterate(x.pyo, state...)
            if res === nothing
                return nothing
            else
                value, state = res
                return value, state
            end
        end

        Base.convert(::Type{$name}, o::PyObject) = $name(o)

        push!(pre_type_map, ($class, $name))
    end
end

function Base.Array(x::PyObject)
    if x.dtype.kind == "M"
        map(z -> unix2datetime(z / 1e9), x.astype("int64"))
    elseif x.dtype.kind == "m"
        map(z -> Millisecond(z / 1e6), x.astype("int64"))
    elseif x.dtype.kind == "O" && get(x, 0) isa String
        convert(Array{String}, x)
    elseif x.dtype.kind == "O"
        map(PyAny, PyAny(x))
    else
        convert(PyAny, x)
    end
end

Base.Array(x::PandasWrapped) = Array(x."values")

function Base.values(x::PandasWrapped)
    # Zero-copy conversion to a Julia native type is possible
    x_kind = x.dtype.kind
    if x_kind in ["i", "u", "f", "b"]
        pyarray = convert(PyArray, x."values")
        if pyarray.f_contig
            dims, T, ptr = size(pyarray), eltype(pyarray), pyarray.data
            if Int(ptr) % Base.datatype_alignment(T) == 0
                unsafe_wrap(Array, ptr, dims)
            else
                Aflat = unsafe_wrap(Array, Ptr{UInt8}(ptr), prod(dims) * sizeof(T))
                A = reshape(reinterpret(T, Aflat), dims)
            end
        else
            Array(pyarray)
        end
    else
        Array(x)
    end
end

struct StringRange{T <: AbstractString}
    start::T
    stop::T
end
(::Colon)(start::T, stop::T) where T <: AbstractString = StringRange{T}(start, stop)

PyCall.PyObject(x::Colon) = pybuiltin("slice")(nothing, nothing, nothing)
PyCall.PyObject(x::StringRange) = pybuiltin("slice")(x.start, x.stop)

fix_arg(x) = x
fix_arg(x::StepRange) = pybuiltin("slice")(x.start, x.start + length(x) * x.step, x.step)
fix_arg(x::UnitRange) = fix_arg(StepRange(x.start, 1, x.stop))

fix_arg(x, offset) = fix_arg(x)
fix_arg(x::Union{Integer, AbstractArray{<:Integer}}, offset) = offset ? fix_arg(x .- 1) : fix_arg(x)

macro pyasvec(class)
    index_expr = quote
        function Base.getindex(pyt::$class, args...)
            offset = should_offset(pyt, args...)
            new_args = tuple([fix_arg(arg, offset) for arg in args]...)
            new_args = (length(new_args) == 1 ? new_args[1] : new_args)
            pyt.__getitem__(new_args)
        end

        Base.view(pyt::$class, is...) = Base.getindex(x, is...)

        function Base.setindex!(pyt::$class, value, idxs...)
            offset = should_offset(pyt, idxs...)
            new_idx = [fix_arg(idx, offset) for idx in idxs]
            if length(new_idx) > 1
                pyt.__setitem__(tuple(new_idx...), value)
            else
                pyt.__setitem__(new_idx[1], value)
            end
        end
    end

    length_expr = if class in [:Iloc, :Loc]
        :(Base.length(x::$class) = x.obj.__len__())
    else
        :(Base.length(x::$class) = x.__len__())
    end

    quote
        $index_expr
        $length_expr
        Base.lastindex(x::$class) = length(x)
        Base.lastindex(x::$class, i) = size(x, i)
    end
end

@pytype DataFrame () -> pd.core.frame.DataFrame
@pytype Series () -> pd.core.series.Series
@pytype Iloc () -> pd.core.indexing._iLocIndexer
@pytype Loc () -> pd.core.indexing._LocIndexer
@pytype Index () -> pd.core.index.Index
@pytype MultiIndex () -> pd.core.index.MultiIndex

Base.size(x::Union{Loc, Iloc}) = x.obj.shape
Base.size(df::PandasWrapped, i::Integer) = size(df)[i]
Base.size(df::PandasWrapped) = df.shape
Base.ndims(df::PandasWrapped) = length(size(df))

Base.isempty(df::PandasWrapped) = df.empty
Base.empty!(df::PandasWrapped) = df.drop(df.index, inplace = true)

should_offset(::Any, args...) = false
should_offset(::Union{Iloc, Index, MultiIndex}, args...) = true

@pyasvec Series
@pyasvec Loc
@pyasvec Iloc
@pyasvec DataFrame
@pyasvec Index
@pyasvec MultiIndex

Base.show(io::IO, df::PandasWrapped) = println(io, df.__str__())

for (jl_op, py_op, py_opᵒ) in [(:+, :__add__, :__add__), (:*, :__mul__, :__mul__),
                               (:/, :__div__, :__rdiv__), (:-, :__sub__, :__rsub__),
                               (:(==), :__eq__, :__eq__), (:!=, :__ne__, :__ne__),
                               (:>, :__gt__, :__lt__), (:<, :__lt__, :__gt__),
                               (:>=, :__ge__, :__le__), (:<=, :__le__, :__ge__),
                               (:&, :__and__, :__and__), (:|, :__or__, :__or__)]
    @eval begin
        Base.$jl_op(x::PandasWrapped, y) = x.$py_op(y)
        Base.$jl_op(y, x::PandasWrapped) = x.$py_opᵒ(y)
        Base.$jl_op(x::PandasWrapped, y::PandasWrapped) = invoke($jl_op, Tuple{PandasWrapped, Any}, x, y)
    end
end

for (jl_op, py_op) in [(:-, :__neg__), (:!, :__neg__)]
    @eval Base.$jl_op(x::PandasWrapped) = x.$py_op()
end

include("operators_v6.jl")

DataFrame(pairs::Pair...) = DataFrame(Dict(pairs...))

function Base.eltype(s::Series)
    dtype_map = Dict(
        np.dtype("int8") => Int8,
        np.dtype("uint8") => UInt8,
        np.dtype("int16") => Int16,
        np.dtype("uint16") => UInt16,
        np.dtype("int32") => Int32,
        np.dtype("uint32") => UInt32,
        np.dtype("int64") => Int64,
        np.dtype("uint64") => UInt64,
        np.dtype("float16") => Float16,
        np.dtype("float32") => Float32,
        np.dtype("float64") => Float64,
        np.dtype("object") => String,
    )
    get(dtype_map, s.dtype, Any)
end

Base.eltype(df::DataFrame) = Tuple{[eltype(df[c]) for c in df.columns]...}

function Base.map(f::Function, s::Series)
    if eltype(s) <: Real
        Series([f(_) for _ in values(s)])
    else
        Series([f(_) for _ in s])
    end
end

Base.get(df::PandasWrapped, key, default) = df.get(key, default)

Base.getindex(s::Series, c::CartesianIndex{1}) = s[c[1]]
    
Base.copy(df::PandasWrapped) = df.copy()

include("tabletraits.jl")

function DataFrame(obj)
    y = _construct_pandas_from_iterabletable(obj)
    if y === nothing
        return invoke(DataFrame, Tuple{Vararg{Any}}, obj)
    else
        return y
    end
end

Base.getproperty(x::PandasWrapped, s::Symbol) = s == :pyo ? getfield(x, s) : getproperty(x.pyo, s)

Base.getproperty(x::PandasWrapped, s::String) = getproperty(x.pyo, s)

function Base.setproperty!(x::PandasWrapped, s::Symbol, v)
    if s == :pyo
        return setfield!(x, s, v)
    else
        setproperty!(x.pyo, s, v)
    end
end

include("miscellaneous.jl")
include("juno.jl")

end
