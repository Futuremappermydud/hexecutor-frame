import unittest

from model_artifact_policy import embedded_model_array, generic_name, generic_text, is_artifact


class ArtifactPolicyTests(unittest.TestCase):
    def test_legacy_and_generic_artifact_locations(self):
        for root in ('quest-depth', 'depth-model', 'qd'):
            for suffix in ('models/context.bin', 'recovered/graph.json', 'metadata/graph.json',
                           'frame/native_program/fixtures/input.epcpu', 'portable/check/package/run.py'):
                self.assertTrue(is_artifact(f'{root}/{suffix}'))

    def test_raw_weights_and_source_encoded_dumps(self):
        for path in ('other/weights.safetensors', 'other/model.onnx.data', 'other/pool.pool1',
                     'other/arrays.npy', 'other/graph.hxfg', 'other/hand_landmarker.task', 'other/qnn_runlist_primary.generated.h',
                     'other/xwb1022_actual_data.h', 'other/ch5099_padw7_input_data.h'):
            self.assertTrue(is_artifact(path))
        self.assertTrue(is_artifact('depth-model/100000'))

    def test_code_recipes_and_links_remain(self):
        for path in ('hexecutor/models', 'hexecutor/recovered', 'depth-model/training/README.md',
                     'depth-model/tools/prepare_inputs.py', 'hexecutor/frame/qd_memory.c',
                     'hexecutor/frame/native_kernels/make_fixture.py'):
            self.assertFalse(is_artifact(path))

    def test_large_embedded_array(self):
        data = b'static const uint8_t weights[5000] = {' + b'123,' * 5000 + b'};'
        self.assertTrue(embedded_model_array(data))
        self.assertFalse(embedded_model_array(b'static const uint8_t offsets[3]={1,2,3};'))

    def test_rename_preserves_unrelated_request_identifiers(self):
        source = 'Request request requests REQUEST requested requestSequence QuestDepthPipeline questPrepared QUEST_WIDTH FRAME_QUEST_MODEL quest-depth quest_depth.py'
        expected = 'Request request requests REQUEST requested requestSequence ModelDepthPipeline modelPrepared MODEL_WIDTH FRAME_DEPTH_MODEL depth-model model_depth.py'
        self.assertEqual(generic_name(source), expected)

    def test_external_source_urls_stay_valid(self):
        self.assertEqual(generic_text(b'Quest https://example.com/quest/model Quest'),
                         b'Model https://example.com/quest/model Model')


if __name__ == '__main__':
    unittest.main()
