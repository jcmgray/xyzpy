import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from xyzpy.gen import cropping


class TestAtomicCropFiles:
    def test_failed_write_keeps_previous_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.jbdmp"
            cropping.write_to_disk(("old",), path)

            with (
                patch.object(
                    cropping.pickle,
                    "dump",
                    side_effect=RuntimeError("write failed"),
                ),
                pytest.raises(RuntimeError, match="write failed"),
            ):
                cropping.write_to_disk(("new",), path)

            assert cropping.read_from_disk(path) == ("old",)
            assert list(Path(directory).iterdir()) == [path]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_atomic_write_preserves_permissions(self, tmp_path):
        control = tmp_path / "control"
        control.write_bytes(b"")
        path = tmp_path / "data"
        cropping.write_to_disk(1, path)
        assert path.stat().st_mode & 0o777 == control.stat().st_mode & 0o777
        path.chmod(0o664)
        cropping.write_to_disk(2, path)
        assert path.stat().st_mode & 0o777 == 0o664
