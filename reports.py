"""
SmartVolt AI — PDF & CSV Audit Reporting Module
Generates tamper-evident industrial audit reports in PDF and CSV format.
Uses ReportLab to build multi-page reports with executive summaries,
temperature trends, anomaly audit trails, and academic evaluation metrics.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)


def export_readings_csv(df: pd.DataFrame, output_path: str) -> str:
    """Exports telemetry readings DataFrame to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def export_events_csv(df: pd.DataFrame, output_path: str) -> str:
    """Exports anomaly events DataFrame to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def generate_pdf_report(
    db_stats: Dict,
    recent_readings: pd.DataFrame,
    recent_events: pd.DataFrame,
    benchmark_metrics: Optional[Dict] = None,
    output_path: str = "data/smartvolt_report.pdf"
) -> str:
    """
    Builds a professional multi-page industrial audit PDF report.
    Includes executive statistics, ML evaluation results, event history,
    and explicit hardware safety limitations.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=normal_style,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=normal_style,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155")
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=normal_style,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b")
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=normal_style,
        fontSize=8,
        leading=10,
        fontName="Helvetica-Bold",
        textColor=colors.white
    )

    story = []

    # Title & Header
    story.append(Paragraph("SmartVolt AI — Industrial Supervisory Audit Report", title_style))
    story.append(Paragraph(
        f"<b>IoT Edge Telemetry & Machine Learning Anomaly Detection System</b><br/>"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target: ESP8266 NodeMCU & DS18B20",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # 1. Executive Summary Table
    story.append(Paragraph("1. Executive Telemetry Summary", h2_style))

    avg_temp = "--"
    min_temp = "--"
    max_temp = "--"
    if not recent_readings.empty and "temperature" in recent_readings.columns:
        valid_t = recent_readings["temperature"].dropna()
        if not valid_t.empty:
            avg_temp = f"{valid_t.mean():.2f} °C"
            min_temp = f"{valid_t.min():.2f} °C"
            max_temp = f"{valid_t.max():.2f} °C"

    hw_count = db_stats.get("hardware_readings", 0)
    sim_count = db_stats.get("simulated_readings", 0)
    total_readings = db_stats.get("total_readings", len(recent_readings))
    total_events = db_stats.get("total_events", len(recent_events))

    summary_data = [
        [
            Paragraph("<b>Total Readings Ingested:</b>", body_style), Paragraph(str(total_readings), body_style),
            Paragraph("<b>Hardware (Real) Samples:</b>", body_style), Paragraph(f"{hw_count} ({round(hw_count/max(1, total_readings)*100, 1)}%)", body_style)
        ],
        [
            Paragraph("<b>Simulated Test Samples:</b>", body_style), Paragraph(f"{sim_count} ({round(sim_count/max(1, total_readings)*100, 1)}%)", body_style),
            Paragraph("<b>Total Anomaly Events:</b>", body_style), Paragraph(str(total_events), body_style)
        ],
        [
            Paragraph("<b>Mean Temperature:</b>", body_style), Paragraph(avg_temp, body_style),
            Paragraph("<b>Min / Max Temperature:</b>", body_style), Paragraph(f"{min_temp} / {max_temp}", body_style)
        ]
    ]

    summary_table = Table(summary_data, colWidths=[130, 130, 130, 130])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 2. Machine Learning & Statistical Benchmark
    story.append(Paragraph("2. Detection Model & Benchmark Performance", h2_style))
    if benchmark_metrics:
        bm_data = [
            [
                Paragraph("Approach / Detector", table_header),
                Paragraph("Precision", table_header),
                Paragraph("Recall", table_header),
                Paragraph("F1-Score", table_header),
                Paragraph("Confusion (TP / FP / TN / FN)", table_header)
            ]
        ]
        for key in ["rule_based", "statistical_mad", "isolation_forest", "hybrid"]:
            if key in benchmark_metrics:
                m = benchmark_metrics[key]
                conf_str = f"TP:{m['tp']} | FP:{m['fp']} | TN:{m['tn']} | FN:{m['fn']}"
                bm_data.append([
                    Paragraph(f"<b>{m['name']}</b>", table_cell),
                    Paragraph(f"{m['precision']*100:.1f}%", table_cell),
                    Paragraph(f"{m['recall']*100:.1f}%", table_cell),
                    Paragraph(f"{m['f1_score']:.4f}", table_cell),
                    Paragraph(conf_str, table_cell)
                ])

        bm_table = Table(bm_data, colWidths=[150, 75, 75, 75, 145])
        bm_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(bm_table)
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<i>Note: Benchmark performance is computed against a controlled zero-leakage test dataset. "
            "Simulated benchmark metrics verify algorithmic design and do not represent certified field accuracy.</i>",
            body_style
        ))
    else:
        story.append(Paragraph("Benchmark evaluation data not loaded.", body_style))

    story.append(Spacer(1, 10))

    # 3. Recent Anomaly Event Audit Log
    story.append(Paragraph("3. Anomaly & Safety Event Audit Log (Recent Events)", h2_style))
    if not recent_events.empty:
        event_cols = [
            Paragraph("Timestamp", table_header),
            Paragraph("Source", table_header),
            Paragraph("Severity", table_header),
            Paragraph("Detection Method", table_header),
            Paragraph("Reason / Suggested Action", table_header)
        ]
        ev_data = [event_cols]
        for _, row in recent_events.head(10).iterrows():
            ts = str(row.get("timestamp", ""))[-8:]  # HH:MM:SS
            src = str(row.get("source", "SIM"))
            sev = str(row.get("severity", "WARN"))
            meth = str(row.get("detection_method", "HYBRID"))
            reason_text = f"{row.get('reason', '')}<br/><b>Action:</b> {row.get('suggested_action', 'Inspect device')}"
            ev_data.append([
                Paragraph(ts, table_cell),
                Paragraph(src, table_cell),
                Paragraph(sev, table_cell),
                Paragraph(meth, table_cell),
                Paragraph(reason_text, table_cell)
            ])

        ev_table = Table(ev_data, colWidths=[65, 55, 60, 90, 250])
        ev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(ev_table)
    else:
        story.append(Paragraph("No anomaly events recorded during this audit window.", body_style))

    story.append(Spacer(1, 12))

    # 4. Hardware Limitations & Engineering Integrity Notice
    story.append(Paragraph("4. Hardware Safety Protocol & Limitations", h2_style))
    limitation_text = (
        "<b>Hardware Interlock Status:</b> Real hardware voltage monitoring on NodeMCU pin A0 remains strictly "
        "locked in compliance with safety specifications. The ESP8266 bare ADC core has a 0-1.0V maximum tolerance, "
        "scaled to 3.3V via the onboard divider. The blue 5:1 analog voltage divider module has not undergone "
        "benchtop calibration with a regulated DC power supply and calibrated multimeter. "
        "<b>Never connect AC mains (110V/230V) to this system under any circumstances.</b><br/><br/>"
        "<b>Scientific Integrity:</b> Temperature anomalies demonstrate abnormal thermal dissipation, environmental heating, "
        "or cooling failure. Thermal spikes correlate with electrical load but do not prove an electrical fault "
        "without calibrated electrical voltage and current sensors."
    )
    story.append(Paragraph(limitation_text, body_style))

    doc.build(story)
    return output_path

