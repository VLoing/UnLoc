require 'optim'
require 'nn'
require 'image'
require 'cunn'
require 'cudnn'
require 'paths'
local RealPict = require('real_pict')
local t = require 'transforms'

--------------  TO CHANGE --------------------------------
local cmd = torch.CmdLine()
cmd:option('-folder_path', '../UnLoc_Lab_Dataset/',         'Path to dataset : "../UnLoc_Lab_Dataset/"  or "../UnLoc_Adv_Dataset/" or "../UnLoc_Field_Dataset/" ')
local opt = cmd:parse(arg or {})

local folder_path = opt.folder_path
local data_file_path = folder_path .. 'pictures_data.txt'
local labels_fine = require 'labels_fine_estimation'
local model_fine = 'model_fine_estimation.t7' 

---------------- FUNCTIONS -------------------------------
function string.starts(String, Start)
   return string.sub(String,1,string.len(Start))==Start
end


--This function permits to change the score in one clamp reference system to a score in the clamp reference system where the clamp is rotated of 90°
function changeScoreReferentiel(Tensor1D, XorYorRot, angle)
   assert(Tensor1D:dim() == 1 ) 
   local newScore = Tensor1D:clone()
   local size = Tensor1D:size(1)

   if XorYorRot == 'X' and angle == 90 then  --only works if same number of Xclasses and Yclasses and same number of negative classes and positive classes
      for i=1, size do
         newScore[i] = Tensor1D[size + 1 - i]
      end

   elseif XorYorRot == 'Rot' and angle == 90 then 
      for i=1, size do 
         newScore[i] = Tensor1D[(i - 1 - size/2)%size + 1]
      end
   end
   return newScore
end
----------------- PREPROCESSING --------------------

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function PictPreprocess()
      return t.Compose{
         t.CenterCrop(378),  -- 378 = (224/256)*432 
         t.ColorNormalize(meanstd),
         t.Scale(224),
      }
end
------------------CREATE REAL_PICT_DATA DICT----------
--Here we fill the array realPictData. Each realPict object corresponds to one block pose. realPict object is filled with the path of each picture from differents cameras, and with the value (X, Y, Rot) of the block w.r.t. the robot base and w.r.t. the gripper.  

local realPictData = {}

local file = assert(io.open(data_file_path, 'r'))

while true do 
   local line = file:read('*line')
   if not line then break end
   if not string.starts(line, '#') then
      line = line:split(',')

      pictPath = line[1]
      X_abs = tonumber(line[2])
      Y_abs = tonumber(line[3])
      Rot_abs = tonumber(line[4])
      X_tool_abs = tonumber(line[5])
      Y_tool_abs = tonumber(line[6])
      Rot_tool_abs = tonumber(line[7])
      camera_ID = tonumber(line[9])
      pict_number = tonumber(line[10])
      poseID = tonumber(line[11])

      local local_folder = pictPath:split('/')[1]
      local pict_name = pictPath:split('/')[2]

      --the pictures for this fine estimation step are the cropped one obtained with tool_detection.lua in the folder pictures_fine
      local new_local_folder = local_folder .. '_fine'
      local new_pict_name = pict_name:split('%.')[1] .. '_fine.' .. pict_name:split('%.')[2]  
      local pictPath_P = new_local_folder .. '/' .. new_pict_name

      local realPict = realPictData[poseID] and realPictData[poseID] or RealPict(poseID)
      if not realPictData[poseID] then realPictData[poseID] = realPict end
      realPict:setPath(pict_number, camera_ID, pictPath)
      realPict:setL_abs(X_abs, Y_abs, Rot_abs)
      realPict.L_modelPath = pModel_L
      realPict.L_classes = tModel_L
   
      if X_tool_abs ~= nil then 
         realPict:setPath_P(pict_number, camera_ID, pictPath_P)
         realPict:set_tool_abs(pict_number, camera_ID, X_tool_abs, Y_tool_abs, Rot_tool_abs)
         realPict:setP_abs(pict_number, camera_ID)
      end
   end     
end

file:close()


----------------- LOAD MODEL -------------------------


local model = torch.load(model_fine):type('torch.CudaTensor')
model:evaluate()

local X_err = {}
local Y_err = {}
local Rot_err = {}

for k,v in pairs(realPictData) do

   local tot_output_X = nil
   local tot_output_Y = nil
   local tot_output_Rot = nil
   clampPos1 = false
   clampPos2 = false
   --for 2 clamp positions  (pict_number = 3 and pict_number = 5 in pictures_data.txt)
   for _,pictNumber in pairs({3, 5}) do 
      images=nil

      -- for each camera
      for cameraID =0, 2 do 
         local imagePath = realPictData[k]:getPath_P(pictNumber,cameraID) 
         local pictPath = folder_path .. imagePath
         if path.exists(pictPath) then 
            local im = image.load(pictPath, 3, 'float')
            local im = PictPreprocess()(im)
            im = im:view(1,table.unpack(im:size():totable()))
            images = images and torch.cat(images, im, 1) or im
         end
      end

      local output = model:forward(images:cuda())

      if (pictNumber == 3) then 
         tot_output_X = output[1]:sum(1):float():squeeze()
         tot_output_Y = output[2]:sum(1):float():squeeze()
         tot_output_Rot = output[3]:sum(1):float():squeeze()
         clampPos1 = true
      elseif (pictNumber == 5) and clampPos1 then
         tot_output_X = tot_output_X + changeScoreReferentiel(output[2]:sum(1):float():squeeze(), 'X', 90) -- X = -Y
         tot_output_Y = tot_output_Y + changeScoreReferentiel(output[1]:sum(1):float():squeeze(), 'Y', 90) -- Y = X
         tot_output_Rot =  tot_output_Rot + changeScoreReferentiel(output[3]:sum(1):float():squeeze(), 'Rot', 90)
         clampPos2 = true
      end
         
      if clampPos1 and clampPos2 then

         local _, Xidx = tot_output_X:topk(1, 1, true, true)
         local _, Yidx = tot_output_Y:topk(1, 1, true, true)
         local _, Rotidx = tot_output_Rot:topk(1, 1, true, true)
         local Xpred = labels_fine["X"][Xidx:squeeze()]
         local Ypred = labels_fine["Y"][Yidx:squeeze()]
         local Rotpred = labels_fine["Rot"][Rotidx:squeeze()]
         local X_target, Y_target, Rot_target = realPictData[k]:getP_abs(3, 0) -- in the reference system of clamPos1 (i.e. pict_number = 3)

         table.insert(X_err, math.abs(tonumber(Xpred) - tonumber(X_target)))
         table.insert(Y_err, math.abs(tonumber(Ypred) - tonumber(Y_target)))
         table.insert(Rot_err, math.min((tonumber(Rotpred) - tonumber(Rot_target))%180, (tonumber(Rot_target) - tonumber(Rotpred))%180))
         print(math.abs(tonumber(Xpred) - tonumber(X_target)))
      end
   end
end   



local X_err = torch.FloatTensor(X_err)
local Y_err = torch.FloatTensor(Y_err)
local Rot_err = torch.FloatTensor(Rot_err)

local X_ok = torch.le(X_err, 5):float()
local Y_ok = torch.le(Y_err, 5):float()
local Rot_ok = torch.le(Rot_err, 2):float()
local X_Y_ok = torch.cmul(X_ok, Y_ok)
local XYROT_ok = torch.cmul(X_Y_ok, Rot_ok)

x_std = X_err:std()
y_std = Y_err:std()
rot_std = Rot_err:std()

print('% (err X < 5 mm) = ' .. X_ok:mean()*100)
print('% (err Y < 5 mm) = ' .. Y_ok:mean()*100)
print('% (err theta < 2°) = ' .. Rot_ok:mean()*100)
print('% (err X and err Y < 5 mm and theta < 2°) = ' ..XYROT_ok:mean()*100)
print('err X (mm) = ' .. X_err:mean() .. ' +- ' .. x_std)
print('err Y (mm) = ' .. Y_err:mean() .. ' +- ' .. y_std)
print('err Rot (°) = ' .. Rot_err:mean() .. ' +- ' .. rot_std)



