import torch
import pathlib
from model import MattingNetwork
from inference import convert_video

torch.cuda.empty_cache()
if torch.cuda.is_available():
    model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
else:
    model = MattingNetwork('mobilenetv3').eval()

model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))


# "H:\Other computers\My MacBook Air\Movies\benVideoArtProjects\fGroundSuicide.mp4"
# x = pathlib.Path(r'C:\Users\bgenc\Desktop\compOut.mp4')

x = pathlib.PurePath(input('enter the thing '))
fileName = pathlib.PurePath(x).stem
print(fileName)
# x2 = str(x).strip
x2 = pathlib.PurePath(str(x).strip('\"\"'))
y = pathlib.Path(x2.parent, 'fart' + 'test')
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

compFolder = pathlib.Path(y, fileName + 'compOut')
foregroundFolder = pathlib.Path(y, fileName + 'fGround')
alphaTypeFolder = pathlib.Path(y, fileName + 'alphaFolder')

alphaTypeFolder.mkdir(exist_ok=True)
compFolder.mkdir(exist_ok=True)
foregroundFolder.mkdir(exist_ok=True)

if outputType != 'png_sequence':
    alphaType = str(alphaTypeFolder) + '/' + 'alpha.mp4'
    comp = str(compFolder) + '/' + 'compout.mp4'
    foreground = str(foregroundFolder) + '/' + 'fg.mp4'
else:
    alphaType = str(alphaTypeFolder)
    foreground = str(foregroundFolder)
    comp = str(compFolder)

print(foreground)

convert_video(
    model,  # The model, can be on any device (cpu or cuda).
    input_source=str(x2),  # A video file or an image sequence directory.
    output_type=outputType,  # Choose "video" or "png_sequence"
    output_composition=comp,  # File path if video; directory path if png sequence.
    output_alpha=alphaType,  # [Optional] Output the raw alpha prediction.
    output_foreground=foreground,  # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,  # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,  # Process n frames at once for better parallelism.
)
