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

from dataclasses import dataclass
import dataclasses
import dacite
import json
import typing


# The state of a ball in a particular frame.
@dataclass
class BallState:
    # The [x, y] position of the ball.
    position: list # list<float>
        
    # Whether the ball is in freefall.
    freefall: bool


# The labels for a ball in a video.
@dataclass
class BallSequence:        
    # The first frame where the ball appears in the video.
    start_frame: int
        
    # `states[i]` is the state of the ball in frame `start_frame + i` of the video.
    states: typing.List[BallState]
        
    # The color of the ball.
    color: str


# All the labels in a video.
@dataclass
class VideoLabels:
    # The balls in the video.
    balls: typing.Dict[str, BallSequence] # Maps id to labels for the ball. (int: BallSequence)
        
    def next_ball_id(self):
        try:
            return str(max([int(k) for k in self.balls.keys()]) + 1)
        except ValueError:
            return '0'

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return dacite.from_dict(data_class=VideoLabels, data=data)

    def save(self, filename):
        data = dataclasses.asdict(self)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
