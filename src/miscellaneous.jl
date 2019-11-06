export pdhcat, pdvcat, noconsolidation

Base.setindex!(df::DataFrame, v::AbstractArray, col::Union{Symbol, String}) = setindex!(df, v, [col])

function Base.setindex!(df::DataFrame, v::AbstractArray, cols)
    if isempty(df)
        df_merge = DataFrame(v, copy = false, columns = cols)
        df.pyo = df_merge.pyo
    elseif size(v, 1) == length(df) && !issubset(cols, df.columns)
        df_set = DataFrame(v, copy = false, columns = cols, index = df.index)
        df_merge = df.merge(df_set, left_index = true, right_index = true, copy = false)
        df.pyo = df_merge.pyo
    else
        df[cols] = DataFrame(v)
    end
    return df
end

function pdhcat(dfs...)
    dfs = filter(!isempty, collect(dfs))
    df = dfs[1]
    for dfn in dfs[2:end]
        df = df.merge(dfn, left_index = true, right_index = true, copy = false)
    end
    return df
end

pdvcat(xs...) = pd.concat([xs...], axis = 0, ignore_index = true)

function noconsolidation()
    py"""
def _consolidate_inplace(self):
    pass

def _consolidate(self):
    return self.blocks

from pandas.core.internals import BlockManager
BlockManager._consolidate_inplace = _consolidate_inplace
BlockManager._consolidate = _consolidate
"""
end