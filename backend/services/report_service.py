"""
PDF report generation for completed damage assessments.

The report is rendered entirely in memory with
reportlab and returned as bytes. No file is ever
written to disk, and no image, mask array or other
bulk binary is embedded, which keeps each report in
the low kilobytes instead of megabytes.

This module is deliberately free of database and
storage concerns. It receives one complete
assessment record and renders it.
"""

from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_TITLE = "AI Disaster Damage Assessment Report"

DAMAGE_CLASS_NAMES = {
    0: "background",
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}

CLASS_ID_ORDER = [0, 1, 2, 3, 4]

MISSING_VALUE = "N/A"

HEADER_COLOR = colors.HexColor("#1f3b57")
ROW_COLOR = colors.HexColor("#eef3f8")
GRID_COLOR = colors.HexColor("#b9c6d4")
LABEL_COLOR = colors.HexColor("#4a5b6c")

PAGE_MARGIN = 16 * mm


def _display(value: Any) -> str:
    """
    Render any value as readable report text.

    None, empty strings and missing keys collapse to a
    single placeholder so that the layout never breaks
    on partially populated assessments.
    """

    if value is None:
        return MISSING_VALUE

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")

    if isinstance(value, (list, tuple, set)):
        items = [_display(item) for item in value]

        return ", ".join(items) if items else MISSING_VALUE

    text = str(value).strip()

    return text if text else MISSING_VALUE


def _human_size(size_bytes: Any) -> str:
    """
    Format a byte count for the image information table.
    """

    try:
        value = int(size_bytes)
    except (TypeError, ValueError):
        return MISSING_VALUE

    if value < 1024:
        return f"{value} bytes"

    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB ({value} bytes)"

    return f"{value / (1024 * 1024):.2f} MB ({value} bytes)"


def _format_timestamp(value: Any) -> str:
    """
    Format an ISO timestamp for display.
    """

    text = _display(value)

    if text == MISSING_VALUE:
        return text

    return text.replace("T", " ").replace("+00:00", " UTC")


def _first_record(value: Any) -> Dict[str, Any]:
    """
    Normalize an embedded PostgREST resource.

    Embedded one-to-one and one-to-many resources both
    arrive as a list, so accept either shape and always
    return a dictionary.
    """

    if isinstance(value, dict):
        return value

    if isinstance(value, (list, tuple)) and value:
        candidate = value[0]

        if isinstance(candidate, dict):
            return candidate

    return {}


def _image_by_type(
    images: Iterable[Any],
    image_type: str,
) -> Dict[str, Any]:
    """
    Return one image record of the requested type.
    """

    for image in images or []:
        if not isinstance(image, dict):
            continue

        if image.get("image_type") == image_type:
            return image

    return {}


def _dimensions(image: Dict[str, Any]) -> str:
    """
    Format image width and height.
    """

    width = image.get("width")
    height = image.get("height")

    if width is None and height is None:
        return MISSING_VALUE

    return f"{_display(width)} x {_display(height)} px"


def _build_styles() -> Dict[str, ParagraphStyle]:
    """
    Build the paragraph styles used across the report.
    """

    styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            alignment=TA_CENTER,
            textColor=HEADER_COLOR,
            spaceAfter=2 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=LABEL_COLOR,
            spaceAfter=5 * mm,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=HEADER_COLOR,
            spaceBefore=4 * mm,
            spaceAfter=1.6 * mm,
        ),
        "label": ParagraphStyle(
            "CellLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=LABEL_COLOR,
        ),
        "value": ParagraphStyle(
            "CellValue",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
        ),
        "cell": ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
        ),
        "cell_head": ParagraphStyle(
            "TableHead",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
        ),
    }


def _info_table(
    rows: List[tuple],
    styles: Dict[str, ParagraphStyle],
) -> Table:
    """
    Build a two column label and value table.
    """

    data = [
        [
            Paragraph(str(label), styles["label"]),
            Paragraph(_display(value), styles["value"]),
        ]
        for label, value in rows
    ]

    table = Table(
        data,
        colWidths=[52 * mm, 124 * mm],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), ROW_COLOR),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID_COLOR),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    return table


def _statistics_table(
    assessment: Dict[str, Any],
    styles: Dict[str, ParagraphStyle],
) -> Table:
    """
    Build the damage class statistics table.

    All five damage classes are always listed, even when
    the assessment has no stored row for one of them.
    """

    stored: Dict[int, Dict[str, Any]] = {}

    for statistic in assessment.get(
        "assessment_class_statistics"
    ) or []:
        if not isinstance(statistic, dict):
            continue

        try:
            class_id = int(statistic.get("class_id"))
        except (TypeError, ValueError):
            continue

        stored[class_id] = statistic

    header = ["Class ID", "Class name", "Pixel count", "Percentage"]

    body: List[List[Any]] = [
        [
            Paragraph(text, styles["cell_head"])
            for text in header
        ]
    ]

    for class_id in CLASS_ID_ORDER:
        statistic = stored.get(class_id, {})

        class_name = statistic.get(
            "class_name"
        ) or DAMAGE_CLASS_NAMES[class_id]

        percentage = statistic.get("percentage")

        if percentage is None:
            percentage_text = MISSING_VALUE
        else:
            try:
                percentage_text = f"{float(percentage):.2f}%"
            except (TypeError, ValueError):
                percentage_text = _display(percentage)

        body.append(
            [
                Paragraph(str(class_id), styles["cell"]),
                Paragraph(
                    _display(class_name),
                    styles["cell"],
                ),
                Paragraph(
                    _display(statistic.get("pixel_count")),
                    styles["cell"],
                ),
                Paragraph(percentage_text, styles["cell"]),
            ]
        )

    table = Table(
        body,
        colWidths=[20 * mm, 56 * mm, 44 * mm, 32 * mm],
        hAlign="LEFT",
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    HEADER_COLOR,
                ),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_COLOR]),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID_COLOR),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    return table


def generate_assessment_report(
    assessment: Dict[str, Any],
) -> bytes:
    """
    Render one complete assessment as a PDF report.

    The assessment dictionary is the structure returned by
    get_assessment_by_id(). Every field is treated as
    optional so that a partially populated assessment still
    produces a valid report.

    Returns the PDF bytes. Nothing is written to disk.
    """

    if not isinstance(assessment, dict):
        raise ValueError("assessment must be a dictionary")

    styles = _build_styles()
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
        title=REPORT_TITLE,
        author="AI Disaster Damage Assessment",
        subject="Disaster damage assessment report",
        pageCompression=1,
    )

    images = assessment.get("assessment_images") or []
    model = _first_record(assessment.get("assessment_models"))
    prediction = _first_record(assessment.get("assessment_predictions"))

    before_image = _image_by_type(images, "before")
    after_image = _image_by_type(images, "after")

    story: List[Any] = [
        Paragraph(REPORT_TITLE, styles["title"]),
        Paragraph(
            "Generated automatically by the AI Disaster Damage "
            "Assessment backend. This report contains text and "
            "tabular data only; no source imagery is embedded.",
            styles["subtitle"],
        ),
        Paragraph("1. Assessment information", styles["heading"]),
        _info_table(
            [
                ("Assessment ID", assessment.get("id")),
                ("Created date/time", _format_timestamp(assessment.get("created_at"))),
                ("Status", assessment.get("status")),
                ("Damage level", assessment.get("damage_level")),
                ("Damage percentage", assessment.get("damage_percentage")),
                ("Total pixels", assessment.get("total_pixels")),
                ("Damage pixels", assessment.get("damage_pixels")),
            ],
            styles,
        ),
        Paragraph("2. Model information", styles["heading"]),
        _info_table(
            [
                ("Model name", model.get("name")),
                ("Architecture", model.get("architecture")),
                ("Checkpoint", model.get("checkpoint")),
                ("Input channels", model.get("input_channels")),
                ("Output classes", model.get("output_classes")),
                ("Epoch", model.get("epoch")),
                ("Validation loss", model.get("validation_loss")),
            ],
            styles,
        ),
        Paragraph("3. Image information", styles["heading"]),
        _info_table(
            [
                ("Before image filename", before_image.get("filename")),
                ("Before image dimensions", _dimensions(before_image)),
                ("Before image size", _human_size(before_image.get("size_bytes"))),
                ("After image filename", after_image.get("filename")),
                ("After image dimensions", _dimensions(after_image)),
                ("After image size", _human_size(after_image.get("size_bytes"))),
            ],
            styles,
        ),
        Paragraph("4. Prediction information", styles["heading"]),
        _info_table(
            [
                (
                    "Mask dimensions",
                    f"{_display(prediction.get('mask_width'))} x "
                    f"{_display(prediction.get('mask_height'))} px",
                ),
                (
                    "Predicted classes",
                    prediction.get("predicted_classes"),
                ),
            ],
            styles,
        ),
        Paragraph("5. Damage class statistics", styles["heading"]),
        _statistics_table(assessment, styles),
        Spacer(1, 4 * mm),
        Paragraph(
            "Storage path: "
            f"{_display(assessment.get('report_storage_path'))}",
            styles["subtitle"],
        ),
    ]

    document.build(story)

    return buffer.getvalue()
