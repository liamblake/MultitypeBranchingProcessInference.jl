const FIGURE_FILE_EXT = "pdf"
default(; fontfamily="Bookman")

const COLOUR_MAP = Dict(
    1 => :grey,
    2 => :blue,
    3 => :red,
    4 => :green,
)
cmap(i) = (i in keys(COLOUR_MAP)) ? COLOUR_MAP[i] : error("Unknown colour id, expected one of $(keys(COLOUR_MAP)), got $(i)")
pmap(i) = (i in keys(COLOUR_MAP)) ? Symbol(COLOUR_MAP[i], :s) : error("Unknown colour id, expected one of $(keys(COLOUR_MAP)), got $(i)")

const STYLE_MAP = Dict(
    1 => :solid,
    2 => :dash,
    3 => :dot,
)
smap(i) = (i in keys(STYLE_MAP)) ? STYLE_MAP[i] : error("Unknown style id, expected one of $(keys(STYLE_MAP)), got $(i)")
