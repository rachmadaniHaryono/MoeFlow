# -*- coding: utf-8 -*-
import pytest

from moeflow.cmds.main import hello_world


@pytest.mark.asyncio
async def test_hello_world():
    res = await hello_world(object())
    assert res
