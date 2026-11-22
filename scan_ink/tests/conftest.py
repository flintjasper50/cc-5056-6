"""pytest 共享夹具：构造三重层场景与生产提取结果。

夹具只负责准备数据，不含任何期望值——期望值在各测试内从独立参照
（triple_layer 的超采样重合成 / 解析覆盖）现场推导，避免默认值遮错。
"""
from __future__ import annotations

import pytest

from inkline.model import Params
from inkline.pipeline import Project
from inkline.triple_layer import TripleSpec, build_triple_scene


@pytest.fixture(scope="module")
def scene():
    return build_triple_scene(TripleSpec(seed=23))


@pytest.fixture(scope="module")
def extracted(scene):
    """生产链路对三重层场景的墨线分离结果（被测对象）。"""
    proj = Project(image=scene["image"], params=Params())
    return proj, proj.evaluate()


@pytest.fixture(scope="module")
def extracted_gt_paper(scene):
    """给定真值纸色的分离结果，隔离纸色场估计误差。"""
    proj = Project(image=scene["image"], paper_override=scene["paper_gt"],
                   params=Params())
    return proj, proj.evaluate()
