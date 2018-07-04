require 'optim'
require 'nn'
require 'image'
require 'cunn'
require 'cudnn'
require 'paths'
local RealPict = require('real_pict')
local t = require 'transforms'

--------------  TO CHANGE --------
local cmd = torch.CmdLine()
cmd:option('-folder_path', '../UnLoc_Lab_Dataset/',         'Path to dataset : "../UnLoc_Lab_Dataset/"  or "../UnLoc_Adv_Dataset/" or "../UnLoc_Field_Dataset/" ')
cmd:option('-save_pict', false, 'save crop pict of the gripper')
local opt = cmd:parse(arg or {})

local savePict = opt.save_pict
local folder_path = opt.folder_path
local data_file_path = folder_path .. 'pictures_data.txt'

local labels_tool_detection = require 'labels_tool_detection'
local model_tool_detection = 'model_tool_detection.t7' 

---------------- FUNCTIONS ------------------------
function string.starts(String, Start)
   return string.sub(String,1,string.len(Start))==Start
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
         t.Scale(224),
         t.ColorNormalize(meanstd),
      }
end

function CropPict(im, Xc, Yc, cropSize)  -- im:size() = nChannel x height x width  im.crop(src, x1, y1, x2, y2 ) is zero-based
   assert(cropSize <= im:size(3), "cropSize > image width")
   assert(cropSize <= im:size(2),  "cropSize > image height")
   local x1 = math.min(math.max(0, Xc-math.ceil(cropSize/2)), im:size(3) - cropSize) --if Xc near to the left border or right border
   local y1 = math.min(math.max(0, Yc-math.ceil(cropSize/2)), im:size(2) - cropSize)
   local cropIm = image.crop(im, x1, y1, x1 + cropSize, y1 + cropSize)
   return cropIm
end

function DrawRect(im, Xc, Yc, cropSize)
   assert(cropSize <= im:size(3), "cropSize > image width")
   assert(cropSize <= im:size(2),  "cropSize > image height")
   local x1 = math.min(math.max(0, Xc-math.ceil(cropSize/2)), im:size(3) - cropSize) --if Xc near to the left border or right border
   local y1 = math.min(math.max(0, Yc-math.ceil(cropSize/2)), im:size(2) - cropSize)
   local rectIm = image.drawRect(im, x1, y1, x1 + cropSize, y1 + cropSize, {color = {255, 0, 0}, lineWidth = 4})
   return rectIm
end

------------------CREATE REAL_PICT_DATA DICT----------
--Here we fill the array realPictData. Each realPict object corresponds to one block pose. realPict object is filled with the path of each picture from differents cameras, and with the value (X, Y, Rot) of the block w.r.t. the robot base.  
local file = assert(io.open(data_file_path, 'r'))
local realPictData = {}


while true do 
   local line = file:read('*line')
   if not line then break end
   if not string.starts(line, '#') then
      line = line:split(',')

      pictPath = line[1]
      X_abs = tonumber(line[2])
      Y_abs = tonumber(line[3])
      Rot_abs = tonumber(line[4])
      camera_ID = tonumber(line[9])
      pict_number = tonumber(line[10])
      poseID = tonumber(line[11])
      
      local realPict = realPictData[poseID] and realPictData[poseID] or RealPict(poseID)
      if not realPictData[poseID] then realPictData[poseID] = realPict end
      realPict:setPath(pict_number, camera_ID, pictPath)
      realPict:setL_abs(X_abs, Y_abs, Rot_abs)
      realPict.L_modelPath = pModel_L
      realPict.L_classes = tModel_L
   end     
end

file:close()

----------------- LOAD MODEL -------------------------

local model = torch.load(model_tool_detection):type('torch.CudaTensor')
model:evaluate()

local X_err = {}
local Y_err = {}

for k,v in pairs(realPictData) do
   for cameraID=0, 2 do --for each camera
      for _,pictNumber in pairs({3, 5}) do --for 2 clamp positions  (pict_number = 3 and pict_number = 5 in pictures_data.txt)
         local imagePath = realPictData[k]:getPath(pictNumber,cameraID) 

         local local_folder = imagePath:split('/')[1]
         local pict_name = imagePath:split('/')[2]

         local new_local_folder = local_folder .. '_fine'
         local new_pict_name = pict_name:split('%.')[1] .. '_fine.' .. pict_name:split('%.')[2]  

         local path = assert(path.exists(folder_path .. imagePath))
         local im_real = image.load(path, 3, 'float')
         local im = PictPreprocess()(im_real)
         images = im

         local output = model:forward(images:cuda())

         local _, Xidx = output[1]:sum(1):max(2)
         local _, Yidx = output[2]:sum(1):max(2)
         local Xpred = labels_tool_detection["X"][Xidx:squeeze()]
         local Ypred = labels_tool_detection["Y"][Yidx:squeeze()]
   
         local Xc = im_real:size(3)*tonumber(Xpred)
         local Yc = im_real:size(2)*(tonumber(Ypred)+0.047)
          

         local im_crop1 = CropPict(im_real, Xc, Yc, 432)  --432 = 0.4*1080

         local new_pict_path = folder_path .. new_local_folder .. '/' .. new_pict_name
         if savePict then 
            image.save(new_pict_path, im_crop1)
         end
      end
   end
end





