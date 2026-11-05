"""生成按时间排列、保留完整画面的四格联系表。"""
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


def create_sheet(frames):
    canvas=Image.new('RGB',(1024,640),'#202522')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.load_default()
    for i,(ms, data) in enumerate(frames):
        image=Image.open(BytesIO(data)).convert('RGB')
        image.thumbnail((504,280))
        x=(i%2)*512 + (512-image.width)//2
        y=(i//2)*320 + 30 + (280-image.height)//2
        canvas.paste(image,(x,y))
        draw.text(((i%2)*512+8,(i//2)*320+8), f'{i+1}  {ms/1000:.3f}s', fill='white',font=font)
    out=BytesIO();canvas.save(out,format='JPEG',quality=90)
    return out.getvalue()
