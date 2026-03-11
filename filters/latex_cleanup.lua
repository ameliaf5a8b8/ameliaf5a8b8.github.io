fig_count = 0

function Figure(fig)

  fig_count = fig_count + 1

  -- find image
  local img = nil
  for _,block in ipairs(fig.content) do
    if block.t == "Para" or block.t == "Plain" then
      for _,inline in ipairs(block.content) do
        if inline.t == "Image" then
          img = inline
        end
      end
    elseif block.t == "Image" then
      img = block
    end
  end

  if not img then
    return nil
  end

  -- extract filename only
  local src = img.src
  local filename = src:match("([^/]+)$")  -- ucb_reward.svg

  -- build light/dark paths
  local light = "../blog_imgs/light/" .. filename
  local dark  = "../blog_imgs/dark/" .. filename

  -- preserve caption formatting (including math)
  local caption = ""
  if fig.caption and fig.caption.long then
    local tmp = pandoc.Pandoc(fig.caption.long)
    caption = pandoc.write(tmp, "markdown")
    caption = caption:gsub("^%s*", ""):gsub("%s*$", "")
  end

  local id = fig.identifier ~= "" and fig.identifier or ("fig:" .. fig_count)

  local html = [[
<figure id="]] .. id .. [[">
  <img class="light figure-img"
       src="]] .. light .. [["
       alt="]] .. caption .. [[">

  <img class="dark figure-img"
       src="]] .. dark .. [["
       alt="]] .. caption .. [[">

  <figcaption style="text-align:center;">
    <strong>Figure ]] .. fig_count .. [[:</strong> ]] .. caption .. [[
  </figcaption>
</figure>
]]

  return pandoc.RawBlock("html", html)

end