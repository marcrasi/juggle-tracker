# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

import juggle_tracker.labels as labels


def test_serialization():
    example = labels.VideoLabels(
        balls={
            '0': labels.BallSequence(
                start_frame=1337,
                states=[
                    labels.BallState(
                        position=[4, 5],
                        freefall=True),
                    labels.BallState(
                        position=[7, 8],
                        freefall=False)
                ],
                color='orange'),
            '1': labels.BallSequence(
                start_frame=42,
                states=[],
                color='orange'),
        })

    outfile_path = tempfile.mkstemp()[1]
    example.save(outfile_path)
    loaded_example = labels.VideoLabels.load(outfile_path)
    assert(loaded_example == example)
