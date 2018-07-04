require 'optim'
require 'nn'
require 'image'
require 'cunn'
require 'cudnn'
require 'paths'
local RealPict = require('real_pict')
local t = require 'transforms'

--------------  TO CHANGE -------------------------------
local cmd = torch.CmdLine()
cmd:option('-folder_path', '../UnLoc_Lab_Dataset/',         'Path to dataset : "../UnLoc_Lab_Dataset/"  or "../UnLoc_Adv_Dataset/" or "../UnLoc_Field_Dataset/" ')
local opt = cmd:parse(arg or {})

local folder_path = opt.folder_path
local data_file_path = folder_path .. 'pictures_data.txt'
local labels_coarse = require 'labels_coarse_estimation'
local model_coarse = 'model_coarse_estimation.t7' 

---------------- FUNCTIONS ------------------------------
function string.starts(String, Start)
   return string.sub(String,1,string.len(Start))==Start
end

----------------- PREPROCESSING -------------------------

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
         t.Scale(256),
         t.ColorNormalize(meanstd),
         t.CenterCrop(224),
      }
end

------------------CREATE REAL_PICT_DATA DICT----------
--Here we fill the array realPictData. Each realPict object corresponds to one block pose. realPict object is filled with the path of each picture from differents cameras, and with the value (X, Y, Rot) of the block w.r.t. the robot base. 

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
      camera_ID = tonumber(line[9])
      pict_number = tonumber(line[10])
      poseID = tonumber(line[11])

      local realPict = realPictData[poseID] and realPictData[poseID] or RealPict(poseID)
      if not realPictData[poseID] then realPictData[poseID] = realPict end
      realPict:setPath(pict_number, camera_ID, pictPath)
      realPict:setL_abs(X_abs, Y_abs, Rot_abs)
      realPict.L_modelPath = model_coarse
      realPict.L_classes = labels_coarse
   end     
end
file:close()

----------------- LOAD MODEL -------------------------

local model = torch.load(model_coarse):type('torch.CudaTensor')
model:evaluate()

local X_err = {}
local Y_err = {}
local Rot_err = {}

-- for each block pose
for k,v in pairs(realPictData) do 
   images=nil
   -- for picture number 1 (i.e. with robot arm in random pose)
   for _,pictNumber in pairs({1}) do
      --for each cameras (change the value of cameras here to have the results just for one camera, without results aggregation)
      for cameraId=0, 2 do  
         local imagePath = realPictData[k]:getPath(pictNumber,cameraId) 
         local path = assert(path.exists(folder_path .. imagePath))
         local im = image.load(path, 3, 'float')
         local im = PictPreprocess()(im)
         im = im:view(1,table.unpack(im:size():totable()))
         images = images and torch.cat(images, im, 1) or im
      end
   end

  
   local output = model:forward(images:cuda())

   local _, Xidx = output[1]:sum(1):max(2)
   local _, Yidx = output[2]:sum(1):max(2)
   local _, Rotidx = output[3]:sum(1):max(2)
   local Xpred = labels_coarse["X"][Xidx:squeeze()]
   local Ypred = labels_coarse["Y"][Yidx:squeeze()]
   local Rotpred = labels_coarse["Rot"][Rotidx:squeeze()]
   
   local Xtarget = tonumber(v.L_abs.X)
   local Ytarget = tonumber(v.L_abs.Y)
   local Rottarget = tonumber(v.L_abs.Rot)

   table.insert(X_err, math.abs(tonumber(Xpred) - Xtarget ))
   table.insert(Y_err, math.abs(tonumber(Ypred) - Ytarget ))
   table.insert(Rot_err, math.min((tonumber(Rotpred) - Rottarget)%180, (Rottarget - tonumber(Rotpred))%180))
   print(math.abs(tonumber(Xpred) - Xtarget))

end

   
X_err = torch.FloatTensor(X_err)
Y_err = torch.FloatTensor(Y_err)
Rot_err = torch.FloatTensor(Rot_err)

local X_ok = torch.le(X_err, 60):float()
local Y_ok = torch.le(Y_err, 60):float()
local Rot_ok = torch.le(Rot_err, 10):float()
local X_Y_ok = torch.cmul(X_ok, Y_ok)

print( '% (err X < 60 mm) = ' .. X_ok:mean()*100)
print('% (err Y < 60 mm) = ' .. Y_ok:mean()*100)
print('% (err theta < 10°) = ' .. Rot_ok:mean()*100)
print('% (err X and err Y < 60 mm) = ' ..X_Y_ok:mean()*100)





