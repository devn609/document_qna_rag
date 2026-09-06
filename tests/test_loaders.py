from pathlib import Path
import csv

import pytest

from app.loaders import load_file, load_directory

def test_load_txt(tmp_path):
    """TXT files should be loaded with their original content."""
    file = tmp_path / "document.txt"

    sample_text = "Testing QnA"
    file.write_text(sample_text, encoding="utf-8")

    result = load_file(file)

    assert result == sample_text


def test_load_markdown(tmp_path):
    """Markdown files should be loaded as text."""
    file = tmp_path / "document.md"
    file.write_text("# RAG Bot\n\nThis is a document.", encoding="utf-8")

    result = load_file(file)

    assert "# RAG Bot" in result
    assert "This is a document." in result


def test_load_csv(tmp_path):
    """CSV rows should be converted into searchable text."""
    file = tmp_path / "employees.csv"

    with file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Department"])
        writer.writerow(["Alex", "Engineering"])
        writer.writerow(["Marcus", "Marketing"])
        writer.writerow(["Ryan", "Sales"])

    result = load_file(file)

    assert "Name | Department" in result
    assert "Alex | Engineering" in result
    assert "Marcus | Marketing" in result
    assert "Ryan | Sales" in result


def test_load_empty_file(tmp_path):
    """An empty text file should return an empty string."""
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")

    result = load_file(file)

    assert result == ""


def test_unsupported_file_type(tmp_path):
    """Unsupported extensions should raise ValueError."""
    file = tmp_path / "document.xyz"
    file.write_text("some content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(file)
