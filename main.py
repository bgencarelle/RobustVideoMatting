import torch
import pathlib
torch.cuda.empty_cache()

from model import MattingNetwork

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
from inference import convert_video

#x = pathlib.Path(r'C:\Users\bgenc\Desktop\compOut.mp4')

x = pathlib.PurePath(input('enter the thing '))
fileName = pathlib.PurePath(x).stem
print(fileName)
#x2 = str(x).strip
x2 = pathlib.PurePath(str(x).strip('\"\"'))
y = pathlib.Path(x2.parent, 'fart')
y.mkdir(exist_ok=True)

print(x2)
print(y)

outputType = 'png'
a = 0
prompt2 = 'png'
while a != 1:
    prompt2 = input('png or video')
    if prompt2 == 'png':
        outputType = 'png_sequence'
        a = 1
    elif prompt2 == 'video':
        outputType = 'video'
        a = 1
    else:
        print('we are all busy here: PNG OR VIDEO')

print('your source is', x3)

if outputType =='png_sequence':
    comp = 'compOut'
    foreground = 'fGround'
    alphaType = 'alphaFolder'
else:
    alphaType = 'alpha.mp4'
    comp = 'compOut.mp4'
    foreground = 'fGround.mp4'

convert_video(
    model,  # The model, can be on any device (cpu or cuda).
    input_source=x3,  # A video file or an image sequence directory.
    output_type=outputType,  # Choose "video" or "png_sequence"
    output_composition=comp,  # File path if video; directory path if png sequence.
    output_alpha=alphaType,  # [Optional] Output the raw alpha prediction.
    output_foreground=foreground,  # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,  # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,  # Process n frames at once for better parallelism.
)
