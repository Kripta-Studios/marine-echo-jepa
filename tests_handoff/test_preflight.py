"""No network and no external-account access in these tests."""
import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from tools.probe_data import validate_url
from tools.verify_handoff import verify_manifest

ROOT = Path(__file__).resolve().parents[1]

class AccessPolicyTests(unittest.TestCase):
    def test_allowed(self): validate_url('https://download.pangaea.de/data.zip', {'download.pangaea.de'})
    def test_http_rejected(self):
        with self.assertRaises(ValueError): validate_url('http://download.pangaea.de/data.zip', {'download.pangaea.de'})
    def test_unknown_host_rejected(self):
        with self.assertRaises(ValueError): validate_url('https://example.org/data.zip', {'download.pangaea.de'})
    def test_credentials_rejected(self):
        with self.assertRaises(ValueError): validate_url('https://a:b@download.pangaea.de/data.zip', {'download.pangaea.de'})
    def test_nonstandard_port_rejected(self):
        with self.assertRaises(ValueError): validate_url('https://download.pangaea.de:8080/data.zip', {'download.pangaea.de'})

class ManifestTests(unittest.TestCase):
    def test_good_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'a.txt').write_bytes(b'abc')
            digest=hashlib.sha256(b'abc').hexdigest()
            (root/'sums').write_text(f'{digest}  a.txt\n')
            self.assertEqual(verify_manifest(root,root/'sums'),1)
    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'a.txt').write_bytes(b'def')
            digest=hashlib.sha256(b'abc').hexdigest()
            (root/'sums').write_text(f'{digest}  a.txt\n')
            with self.assertRaises(ValueError): verify_manifest(root,root/'sums')
    def test_escaping_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'sums').write_text('0'*64+'  ../outside\n')
            with self.assertRaises(ValueError): verify_manifest(root,root/'sums')

class ConfigurationTests(unittest.TestCase):
    def test_agent_role_bindings_match(self):
        roles=json.loads((ROOT/'configs/agent_roles.json').read_text())
        for name, expected in roles['subagents'].items():
            with (ROOT/f'.codex/agents/{name}.toml').open('rb') as f: actual=tomllib.load(f)
            self.assertEqual(actual['name'],name)
            self.assertEqual(actual['model'],expected['model'])
            self.assertEqual(actual['model_reasoning_effort'],expected['reasoning_effort'])
            self.assertTrue(actual['developer_instructions'])
    def test_budget_has_no_paid_authorization(self):
        budget=json.loads((ROOT/'configs/budget.json').read_text())
        self.assertFalse(budget['cloud_provisioning_authorized'])
        self.assertEqual(sum(budget['allocation_ceiling'].values()),500)
    def test_no_scientific_results_in_initial_tasks(self):
        tasks=json.loads((ROOT/'orchestration/tasks.json').read_text())['tasks']
        self.assertTrue(all(task['status']=='TODO' for task in tasks))
    def test_spec_parameters_match(self):
        config=json.loads((ROOT/'configs/mvp.json').read_text())
        protocol=json.loads((ROOT/'configs/experiments.json').read_text())
        self.assertEqual(sum(protocol['partitions'].values()),1.0)
        self.assertEqual(config['prediction']['horizons_hours'],[1,3,6])
        self.assertEqual(config['prediction']['quantiles'],protocol['quantiles'])
