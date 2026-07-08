"""Unit tests for wrappers/wrapper_utils.py's pure helpers: build_openvla_prompt,
center_crop_and_resize. Both are model-agnostic (no torch/GPU call), so they import and run
directly against the real module.
"""

from PIL import Image

from wrappers.wrapper_utils import build_openvla_prompt, center_crop_and_resize


def test_build_openvla_prompt_default_format_lowercases_task():
    prompt = build_openvla_prompt("Pick Up The Bowl", base_vla_name="openvla-7b")
    assert prompt == "In: What action should the robot take to pick up the bowl?\nOut:"


def test_build_openvla_prompt_v01_uses_system_prompt_format():
    prompt = build_openvla_prompt("pick up the bowl", base_vla_name="openvla-v01-7b")
    assert prompt.startswith("A chat between a curious user")
    assert "USER: What action should the robot take to pick up the bowl? ASSISTANT:" in (
        prompt
    )


def test_build_openvla_prompt_default_base_name_uses_default_format():
    prompt = build_openvla_prompt("close the drawer")
    assert prompt == "In: What action should the robot take to close the drawer?\nOut:"


def test_center_crop_and_resize_output_size_matches_requested_size():
    image = Image.new("RGB", (256, 256))
    result = center_crop_and_resize(image, crop_scale=0.9, size=224)
    assert result.size == (224, 224)


def test_center_crop_and_resize_full_crop_scale_is_near_identity_size():
    image = Image.new("RGB", (100, 100))
    result = center_crop_and_resize(image, crop_scale=1.0, size=100)
    assert result.size == (100, 100)


def test_center_crop_and_resize_linear_scale_is_sqrt_of_crop_scale():
    # crop_scale=0.25 (area) -> linear_scale=0.5 -> crop box is half the original
    # width/height (100x100 out of 200x200), centered on (50,50)-(150,150). Use a
    # single-color center square to make the crop boundary observable after resize:
    # a solid red 100x100 center square on a blue background should resize to solid red.
    image = Image.new("RGB", (200, 200), color=(0, 0, 255))
    for x in range(50, 150):
        for y in range(50, 150):
            image.putpixel((x, y), (255, 0, 0))

    result = center_crop_and_resize(image, crop_scale=0.25, size=50)

    assert result.size == (50, 50)
    assert result.getpixel((25, 25)) == (255, 0, 0)
