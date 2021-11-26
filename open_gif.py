import pyglet


def run_gif(gif_path='out_gif'):
    gif = pyglet.image.load_animation(gif_path)
    gif_sprite = pyglet.sprite.Sprite(gif)
    w = gif_sprite.width
    h = gif_sprite.height
    win = pyglet.window.Window(width=w, height=h)

    @win.event
    def on_draw():
        win.clear()
        gif_sprite.draw()

    pyglet.app.run()
