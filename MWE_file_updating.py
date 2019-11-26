classes = ['asdf', 'asdf2', 'asdf3']
#with open("yolov3.cfg", 'r+') as fp:
with open("yolov3.cfg", 'r') as f:
    lines = f.readlines()

with open("yolov3.cfg", 'w') as f:
    for i, line in enumerate(lines):
        if i == 2:
            line = "filters=" + str((len(classes) + 5)*3) + "\n"
        f.write(line)
