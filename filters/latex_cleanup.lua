-- latex_cleanup.lua
local function strip_refs(text)
  if not text then return text end
  text = text:gsub("\\label%s*%b{}", "")
  text = text:gsub("\\eqref%s*%b{}", "")
  text = text:gsub("\\tag%s*%b{}", "")
  return text
end

local function strip_environment(text, env)
  return text:match("\\begin{" .. env .. "}([%s%S]-)\\end{" .. env .. "}")
end

local function strip_raw(elem)
  if elem.format == "latex" then
    elem.text = strip_refs(elem.text)
    if elem.text == "" then return {} end
    return elem
  end
end

local function fix_braces(text)
  text = text:gsub("\\{", "\\lbrace ")
  text = text:gsub("\\}", " \\rbrace")
  return text
end

local function escape_underscores(text)
  text = text:gsub("([^\\])_", "%1\\_")
  text = text:gsub("^_", "\\_")
  return text
end

-- helper: append DisplayMath with inline separator
local function add_display(blocks, line, is_last)
  line = escape_underscores(line)
  table.insert(blocks, pandoc.Math("DisplayMath", line))
  if not is_last then
    table.insert(blocks, pandoc.Space())
  end
end

local function process_math(math)
  local text = math.text

  -- aligned → multiple display math inlines
  if text:match("\\begin{aligned}") then
    local inner = strip_environment(text, "aligned")
    if inner then inner = inner:gsub("%$%$", "") end
    if not inner then return math end

    local lines = {}
    for line in inner:gmatch("(.-)\\\\%s*\n?") do
      table.insert(lines, line)
    end

    local blocks = {}
    for i, line in ipairs(lines) do
      line = line:gsub("^%s+", ""):gsub("%s+$", "")
      line = strip_refs(line)
      line = line:gsub("&", "")
      line = fix_braces(line)
      add_display(blocks, line, i == #lines)
    end
    return blocks
  end

  -- align → multiple display math inlines
  if text:match("\\begin{align}") then
    local inner = strip_environment(text, "align")
    if inner then inner = inner:gsub("%$%$", "") end
    if not inner then return math end

    local lines = {}
    for line in inner:gmatch("(.-)\\\\%s*\n?") do
      table.insert(lines, line)
    end

    local blocks = {}
    for i, line in ipairs(lines) do
      line = line:gsub("^%s+", ""):gsub("%s+$", "")
      line = strip_refs(line)
      line = line:gsub("&", "")
      line = fix_braces(line)
      add_display(blocks, line, i == #lines)
    end
    return blocks
  end

  -- equation → single display math
  if text:match("\\begin{equation}") then
    local inner = strip_environment(text, "equation")
    if not inner then return math end
    inner = inner:gsub("^%s+", ""):gsub("%s+$", "")
    inner = strip_refs(inner)
    inner = fix_braces(inner)
    inner = escape_underscores(inner)
    return pandoc.Math("DisplayMath", inner)
  end

  -- ordinary inline math
  local cleaned = strip_refs(text)
  cleaned = cleaned:gsub("&", "")
  cleaned = fix_braces(cleaned)
  cleaned = escape_underscores(cleaned)
  math.text = cleaned
  return math
end

return {
  { Math = process_math },
  { RawInline = strip_raw },
  { RawBlock  = strip_raw }
}
