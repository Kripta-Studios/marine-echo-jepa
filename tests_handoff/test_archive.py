"""Metadata guard tests; no claim of completed ingestion/extraction pipeline."""
import io
import stat
import unittest
import zipfile
from tools.archive_guard import validate_zip_metadata

def make_zip(entries):
    target=io.BytesIO()
    with zipfile.ZipFile(target,'w') as z:
        for name,data in entries: z.writestr(name,data)
    target.seek(0)
    return target

class ArchiveTests(unittest.TestCase):
    def test_normal_archive(self):
        with zipfile.ZipFile(make_zip([('data/a.txt',b'abc')])) as z:
            self.assertEqual(validate_zip_metadata(z), {'members':1,'uncompressed_bytes':3})
    def test_parent_traversal(self):
        with zipfile.ZipFile(make_zip([('../outside',b'x')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_absolute_path(self):
        with zipfile.ZipFile(make_zip([('/tmp/outside',b'x')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_windows_drive(self):
        with zipfile.ZipFile(make_zip([('C:/outside',b'x')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_windows_unc(self):
        with zipfile.ZipFile(make_zip([('\\\\host\\share\\x',b'x')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_backslash_traversal(self):
        with zipfile.ZipFile(make_zip([('data\\..\\outside',b'x')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_case_collision(self):
        with zipfile.ZipFile(make_zip([('DATA/a',b'x'),('data/A',b'y')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_reserved_windows_name(self):
        with zipfile.ZipFile(make_zip([('data/CON.txt',b'x')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_file_as_parent(self):
        with zipfile.ZipFile(make_zip([('data',b'x'),('data/a',b'y')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_symlink(self):
        link=zipfile.ZipInfo('link');link.create_system=3;link.external_attr=(stat.S_IFLNK|0o777)<<16
        with zipfile.ZipFile(make_zip([(link,b'../../outside')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z)
    def test_size_limit(self):
        with zipfile.ZipFile(make_zip([('a',b'0123456789')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z,max_bytes=5)
    def test_member_limit(self):
        with zipfile.ZipFile(make_zip([('a',b'x'),('b',b'y')])) as z:
            with self.assertRaises(ValueError): validate_zip_metadata(z,max_members=1)
