import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os

def get_image_path(base_name):
    """
    自动查找文件，支持 png, jpg, jpeg 大小写后缀
    """
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    for ext in extensions:
        full_path = base_name + ext
        if os.path.exists(full_path):
            return full_path
    return None

def create_composite_image(output_filename="setup_composite.png"):
    """
    读取 a, b, c, d 图片并合成组图，自动兼容 JPG/PNG
    """
    
    # 图片的基础名称 (不带后缀)
    base_names = {
        'a': 'a', 'b': 'b',
        'c': 'c', 'd': 'd'
    }

    # 定义每张子图的标题
    titles = {
        'a': '(a) Bolt & Sensor Placement (View 1)',
        'b': '(b) Bolt & Sensor Placement (View 2)',
        'c': '(c) Torque Reading (Example 1)',
        'd': '(d) Torque Reading (Example 2)'
    }

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)

    subplots = [
        (gs[0, 0], 'a'), (gs[0, 1], 'b'),
        (gs[1, 0], 'c'), (gs[1, 1], 'd')
    ]

    try:
        found_all = True
        for pos, key in subplots:
            ax = fig.add_subplot(pos)
            
            # 自动查找图片路径
            img_path = get_image_path(base_names[key])
            
            if img_path is None:
                print(f"❌ Error: Could not find image for '{key}' (tried .png, .jpg, .jpeg)")
                # 创建一个空白图或显示错误文本
                ax.text(0.5, 0.5, f"Missing: {key}", ha='center', va='center')
                found_all = False
            else:
                print(f"Loading: {img_path}")
                img = mpimg.imread(img_path)
                ax.imshow(img)
            
            ax.set_title(titles[key], fontsize=14, y=-0.15)
            ax.axis('off')

        if found_all:
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.1)
            # 如果输入是 jpg，输出通常也建议存为 jpg 或 png
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"✅ Successfully created composite image: {output_filename}")
        else:
            print("⚠️ Some images were missing, composite might be incomplete.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        plt.close(fig)

if __name__ == "__main__":
    create_composite_image()