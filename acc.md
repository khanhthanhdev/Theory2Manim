


# How to Accelerate Video Rendering in Manim

There are several effective ways to speed up video rendering in Manim:

 

## 2. Take Advantage of Multithreaded Frame Writing

Manim version 0.19.0 introduced a significant performance improvement by implementing a separate thread for writing frames to the output stream. This feature allows rendering to continue while frames are being encoded and written to disk.

The implementation uses a queue to pass frames from the main rendering thread to a dedicated writer thread [2](#0-1) 

This is enabled by default and requires no special configuration.

## 3. Use the Caching System

Manim can cache animations to avoid re-rendering parts of a scene that haven't changed. This is especially useful during development when you're iteratively refining specific parts of an animation.

The caching system is enabled by default, but you can disable it with the `--disable_caching` flag if needed. The system works by computing a hash of each animation [3](#0-2) 

## 4. Adjust Quality Settings

You can significantly speed up rendering by using lower quality settings during development:

- `-ql` or `--quality l`: Low quality (854x480, 15fps)
- `-qm` or `--quality m`: Medium quality (1280x720, 30fps)
- `-qh` or `--quality h`: High quality (1920x1080, 60fps)
- `-qp` or `--quality p`: Production quality (2560x1440, 60fps)
- `-qk` or `--quality k`: 4K quality (3840x2160, 60fps)

These quality options affect both resolution and frame rate [4](#0-3) 

You can also set a custom frame rate with the `--fps` flag [5](#0-4) 

## 5. Render Only What You Need

- Use `-s` or `--save_last_frame` to render only the final frame if you don't need the full animation
- Use `-n` flag to render specific animations within a scene, e.g., `-n 3,5` renders only animations 3 through 5 [6](#0-5) 

## 6. Simplify Mobjects When Possible

The more complex your mobjects (shapes, text, etc.), the longer rendering will take. Simplifying your objects can improve performance:

- Use fewer points in shapes
- Use simpler geometries when possible
- Group objects appropriately

Notes:
- The OpenGL renderer may behave slightly differently from the Cairo renderer, so always check your final animations with your intended output renderer.
- The multithreaded frame writing feature was introduced in PR #3888 and is documented in the changelog for version 0.19.0.
- Quality settings primarily affect the output quality, but lower settings generally render faster.
