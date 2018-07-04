local M = {}

local function symTable(j, pas)
   local t = {}
   for i=-j, j do
      if i<0 then 
         table.insert(t, (i+0.5)*pas)
      elseif i>0 then
         table.insert(t, (i-0.5)*pas)
      end
   end
   return t
end

local function signTable(j, pas)
   local t={}
   for i=1, j do
      table.insert(t, (i-0.5)*pas)
   end
   return t
end

M["X"] = symTable(30, 2)
M["Y"] = symTable(30, 2)
M["Rot"] = signTable(90, 2)

return M


