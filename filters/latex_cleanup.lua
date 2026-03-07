fig_count = 0

function get_image(fig)
  for _,block in ipairs(fig.content) do
    if block.t == "Plain" or block.t == "Para" then
      for _,inline in ipairs(block.content) do
        if inline.t == "Image" then
          return inline
        end
      end
    elseif block.t == "Image" then
      return block
    end
  end
  return nil
end

function Figure(fig)

  local img = get_image(fig)
  if not img then
    return nil
  end

  fig_count = fig_count + 1

  local src = img.src
  local base = src:gsub("%.%w+$","")
  base = base:gsub(" ","_")

  local caption = pandoc.utils.stringify(fig.caption)
  local id = fig.identifier or ("fig:" .. fig_count)

  local light = "../" .. base .. "_light.svg"
  local dark  = "../" .. base .. "_dark.svg"

  local html = [[
<figure id="]] .. id .. [[">
  <picture>
    <source srcset="]] .. dark .. [["
            media="(prefers-color-scheme: dark)">
    <img src="]] .. light .. [["
         style="width:100%; display:block; margin:auto;"
         alt="]] .. caption .. [[">
  </picture>
  <figcaption style="text-align:center;">
    <strong>Figure ]] .. fig_count .. [[:</strong> ]] .. caption .. [[
  </figcaption>
</figure>
]]

  return pandoc.RawBlock("html", html)

end