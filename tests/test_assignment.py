from io import BytesIO
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from helper_lib.model import AssignmentCNN
from helper_lib.data_loader import CLASSES
from main import app


def test_specification_shapes_and_parameters():
    model = AssignmentCNN()
    with torch.inference_mode():
        x = torch.zeros(2, 3, 64, 64)
        features = model.features(x)
        assert features.shape == (2, 32, 16, 16)
        assert model(x).shape == (2, 10)
    assert sum(p.numel() for p in model.parameters()) == 825398


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as client:
        yield client


def test_trained_image_inference(client):
    with open('examples/test_image.png', 'rb') as f:
        response = client.post('/classify-image', files={'file': ('test.png', f, 'image/png')})
    assert response.status_code == 200
    result = response.json()
    assert result['class'] in CLASSES
    assert result['class'] == CLASSES[result['class_id']]
    assert abs(sum(result['probabilities'].values()) - 1) < 1e-5


def test_grayscale_conversion(client):
    buffer = BytesIO()
    Image.new('L', (32, 32), color=128).save(buffer, format='PNG')
    response = client.post('/classify-image', files={'file': ('gray.png', buffer.getvalue(), 'image/png')})
    assert response.status_code == 200


@pytest.mark.parametrize('content', [b'', b'this is not an image'])
def test_invalid_images(client, content):
    assert client.post('/classify-image', files={'file': ('bad.png', content)}).status_code == 400


def test_oversized_upload(client):
    assert client.post('/classify-image', files={'file': ('big.png', b'x' * (10*1024*1024+1))}).status_code == 413


def test_missing_upload(client):
    assert client.post('/classify-image').status_code == 422


def test_previous_embedding_routes(client):
    get = client.get('/embedding', params={'word': 'hello'})
    post = client.post('/get-embedding', json={'word': 'hello'})
    assert get.status_code == post.status_code == 200
    assert get.json()['vector_dim'] == 300
    assert get.json()['vector'] == post.json()['vector']
    assert client.post('/get-embedding', json={'word': ' '}).status_code == 400
