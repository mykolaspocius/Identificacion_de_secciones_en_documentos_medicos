from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ClinicalSections(str, Enum):
    PRESENT_ILLNESS = "PRESENT_ILLNESS"
    DERIVED_FROM_TO = "DERIVED_FROM/TO"
    PAST_MEDICAL_HISTORY = "PAST_MEDICAL_HISTORY"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    EXPLORATION = "EXPLORATION"
    TREATMENT = "TREATMENT"
    EVOLUTION = "EVOLUTION"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class SectionAnnotation(BaseModel):
    segment: str
    label: ClinicalSections
    start_offset: int
    end_offset: int

    class Config:
        use_enum_values = True

    def toJson(self):
        return json.dumps(self, ensure_ascii=False, default=lambda o: o.__dict__)

class SectionAnnotations(BaseModel):
    gold: list[SectionAnnotation] = []
    prediction: list[SectionAnnotation] = []

    def toJson(self):
        return json.dumps(self, ensure_ascii=False, default=lambda o: o.__dict__)

class BoundaryAnnotation(BaseModel):
    span: str
    boundary: ClinicalSections | None
    start_offset: int
    end_offset: int

    class Config:
        use_enum_values = True

    def toJson(self):
        return json.dumps(self, ensure_ascii=False,default=lambda o: o.__dict__)

class BoundaryAnnotations(BaseModel):
    gold: list[BoundaryAnnotation] = []
    prediction: list[BoundaryAnnotation] = []

    def toJson(self):
        return json.dumps(self, ensure_ascii=False, default=lambda o: o.__dict__)

class Entry(BaseModel):
    note_id: str
    note_text: str
    section_annotation: SectionAnnotations = SectionAnnotations()
    boundary_annotation: BoundaryAnnotations = BoundaryAnnotations()

    def toJson(self):
        return json.dumps(self, ensure_ascii=False, default=lambda o: o.__dict__)

class ClinAISDataset(BaseModel):
    annotated_entries: dict[str, Entry]
    scores: dict[str, Any] = {}

    def getEntry(self,idx:int) -> Entry:
        key = list(self.annotated_entries.keys())[idx]
        return self.annotated_entries[key]

    def toJson(self):
        return json.dumps(self, ensure_ascii=False, default=lambda o: o.__dict__)