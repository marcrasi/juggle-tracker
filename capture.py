# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2

cap = cv2.VideoCapture(0)
xres = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
yres = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(xres, yres)

i = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    path = 'data\cap6\img\%03d.png' % i
    cv2.imwrite(path, frame)
    i += 1

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

#import torch
#x = torch.rand(5, 3)
#print(x)
#
#print(torch.cuda.is_available())
