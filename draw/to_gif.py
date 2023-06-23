import imageio, os

def png2gif(path):
    png_lst = sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]))
    # print(png_lst)
    frames = []
    for i in png_lst:
        frames.append(imageio.imread(os.path.join(path, i)))
    imageio.mimsave(path+'.gif', frames, 'GIF', duration=0.5)
    print(f'Finish transforming {path}.')


if __name__=="__main__":
    paths = [
        '../pics/h_value_0',
        '../pics/h_value_1',
        '../pics/h_value_2',
        '../pics/h_value_3',
        '../pics/h_value_4',
    ]
    for path in paths:
        png2gif(path)    