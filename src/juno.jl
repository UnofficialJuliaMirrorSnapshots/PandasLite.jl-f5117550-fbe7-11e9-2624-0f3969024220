using Random, Atom

export JTable

mutable struct JTable
    data
end

JTable(x::AbstractArray) = JTable(DataFrame(x))

function Base.open(tbl::JTable)
    csv = joinpath(homedir(), ".cache", randstring() * ".csv")
    tbl.data.to_csv(csv, encoding = "gbk")
    finalizer(x -> rm(csv), tbl)
    Atom.msg("openFile", csv)
end