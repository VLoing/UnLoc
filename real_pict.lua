require 'torch'

local M = {}
local RealPict = torch.class('RealPict', M)

function RealPict:__init(poseID)
   self.poseID = poseID
   self.L_abs = {} --X = , Y =, Rot =  blocks coordinates with respect to the robot base
   self.L_classes = {} 
   self.L_modelPath = ''
   self.line = ''
   self.tool_abs = {}
   self.tool_abs.pict_number = {{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}}}
   self.P_abs = {}
   self.P_abs.pict_number = {{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}},{camera_ID = {[1]={},[2]={},[3]={}}}}
   self.path = {}
   self.path.pict_number = {{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}}}
   self.path_P = {}
   self.path_P.pict_number = {{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}},{camera_ID = {"pathCam0","pathCam1", "pathCam2"}}}
   self.results_L ={}
   self.results_L.pict_number = {{camera_ID = {"resL0","resL1", "resL2"}},{camera_ID = {"resL0","resL1", "resL2"}},{camera_ID = {"resL0","resL1", "resL2"}},{camera_ID = {"resL0","resL1", "resL2"}},{camera_ID = {"resL0","resL1", "resL2"}},{camera_ID = {"resL0","resL1", "resL2"}},{camera_ID = {"resL0","resL1", "resL2"}}}
   self.pose_completed = {0, 0, 0, 0, 0, 0, 0, 0, 0}
end

--pictNumber between 0 and 6, cameraID between 0 and 2 
function RealPict:getPath(pictNumber, cameraID) 
   return self.path.pict_number[pictNumber+1].camera_ID[cameraID+1]
end

--pictNumber between 0 and 6, cameraID between 0 and 2 
function RealPict:getPath_P(pictNumber, cameraID) 
   return self.path_P.pict_number[pictNumber+1].camera_ID[cameraID+1]
end

--pictNumber between 0 and 6, cameraID between 0 and 2 
function RealPict:setPath(pictNumber, cameraID, path) 
   self.path.pict_number[pictNumber+1].camera_ID[cameraID+1] = path 
end

--pictNumber between 0 and 6, cameraID between 0 and 2 
function RealPict:setPath_P(pictNumber, cameraID, path) 
   self.path_P.pict_number[pictNumber+1].camera_ID[cameraID+1] = path 
end

--results as tensor of probability --pictNumber between 0 and 6, cameraID between 0 and 2 
function RealPict:setResults_L(pictNumber, cameraID, results)
   self.results_L.pict_number[pictNumber+1].cameraID[cameraID+1] = results 
end

--pictNumber between 0 and 6, cameraID between 0 and 2 
function RealPict:getResults_L(pictNumber, cameraID, results)
   return self.results_L.pict_number[pictNumber+1].cameraID[cameraID+1]
end
 
function RealPict:setL_abs(X, Y, Rot)
   self.L_abs.X = X
   self.L_abs.Y = Y
   self.L_abs.Rot = Rot
end

function RealPict:set_tool_abs(pictNumber, cameraID, X_tool, Y_tool, Rot_tool)
   self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].X = X_tool
   self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Y = Y_tool
   self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Rot = Rot_tool
end

function RealPict:get_tool_abs(pictNumber, cameraID)
   local x,y,rot
   x = self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].X
   y = self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Y
   rot = self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Rot
   return x, y, rot
end

function RealPict:getP_abs(pictNumber, cameraID)
   local x,y,rot
   x = self.P_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].X
   y = self.P_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Y
   rot = self.P_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Rot
   return x, y, rot
end

--first, setL_abs and set_tool_abs
function RealPict:setP_abs(pictNumber, cameraID) 
   local X_tool = self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].X --X_tool w.r.t. the robot base
   local Y_tool = self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Y --Y_tool w.r.t. the robot base
   local Rot_tool = self.tool_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Rot --Rot_tool w.r.t. the robot base
   local Rot_tool_rad = math.rad(Rot_tool) -- Rot_tool w.r.t. the robot base in radians
   local X_block = self.L_abs.X --X_block w.r.t. the robot base
   local Y_block = self.L_abs.Y --Y_block w.r.t. the robot base
   local Rot_block = self.L_abs.Rot --block rotation w.r.t. the robot base in degree

   local X_block_tool = (X_block -X_tool)*math.cos(Rot_tool_rad) + (Y_block -Y_tool)*math.sin(Rot_tool_rad) --X_block w.r.t. the tool
   local Y_block_tool = -(X_block -X_tool)*math.sin(Rot_tool_rad) + (Y_block -Y_tool)*math.cos(Rot_tool_rad)--Y_block w.r.t. the tool 
   local Rot_block_tool = (Rot_block - Rot_tool) % 180

   self.P_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].X = X_block_tool
   self.P_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Y = Y_block_tool
   self.P_abs.pict_number[pictNumber+1].camera_ID[cameraID+1].Rot = Rot_block_tool
end

return M.RealPict
