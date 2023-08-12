
import unittest
import pickle
from unittest.mock import patch, Mock
from inference_instance.deployment import Deployment

class TestInference(unittest.TestCase):
    
    @patch('inference_instance.deployment.pickle')
    @patch('inference_instance.deployment.ubiops')
    @patch('inference_instance.deployment.urlopen')
    def test_inference(self, urlopen_mock, ubiops_mock, pickle_mock):
        deployment = Deployment('.',{"project": "project_name"})
        model = Mock()
        model.predict.return_value = ['Yes']
        pickle_mock.load.return_value = model
        data = {"day_features": {"Location":"MelbourneAirport"}}

        res = deployment.request(data)

        self.assertTrue(res['will_it_rain_tomorrow'])
