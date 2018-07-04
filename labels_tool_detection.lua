local M = {}

local function posTable(jmin, jmax, pas)
   local t={}
   for i=jmin, jmax do
      table.insert(t, (i-0.5)*pas)
   end
   return t
end

M["X"] = posTable(1, 50, 0.02)
M["Y"] = posTable(1, 50, 0.02)

return M
