"""
Federated Multi-Agent Learning: Serialized Communication Accounting and Ledger.
"""
import json
import time
from typing import Dict, Any, List, Optional
import torch


def calculate_tensor_payload_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Calculates the exact payload size in bytes for a serialized parameter dictionary.
    Includes tensor raw bytes (numel * element_size) and parameter key string metadata.
    """
    total_bytes = 0
    for name, tensor in state_dict.items():
        if torch.is_tensor(tensor):
            total_bytes += tensor.numel() * tensor.element_size()
            total_bytes += len(name.encode("utf-8"))
    return total_bytes


def calculate_json_metadata_bytes(metadata: Dict[str, Any]) -> int:
    """Calculates byte size of JSON-serialized metadata."""
    try:
        return len(json.dumps(metadata).encode("utf-8"))
    except Exception:
        return 0


class CommunicationLedger:
    """
    Immutable ledger of all model transmissions between coordinator and edge clients.
    Distinguishes:
    - Global model download bytes per recipient
    - Client upload bytes
    - Operational protocol framing bytes
    - Failed/dropped transmission attempts
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record_transmission(
        self,
        round_idx: int,
        client_id: str,
        direction: str,  # 'download' or 'upload'
        payload_bytes: int,
        protocol_bytes: int = 128,
        success: bool = True,
        notes: str = "",
    ):
        record = {
            "timestamp": time.time(),
            "round": round_idx,
            "client_id": client_id,
            "direction": direction,
            "payload_bytes": payload_bytes,
            "protocol_bytes": protocol_bytes,
            "total_bytes": payload_bytes + protocol_bytes,
            "success": success,
            "notes": notes,
        }
        self.records.append(record)

    def get_summary(self) -> Dict[str, Any]:
        """Calculates cumulative communication statistics."""
        total_download_payload = 0
        total_download_protocol = 0
        total_upload_payload = 0
        total_upload_protocol = 0
        failed_download_bytes = 0
        failed_upload_bytes = 0

        for r in self.records:
            if r["direction"] == "download":
                if r["success"]:
                    total_download_payload += r["payload_bytes"]
                    total_download_protocol += r["protocol_bytes"]
                else:
                    failed_download_bytes += r["total_bytes"]
            elif r["direction"] == "upload":
                if r["success"]:
                    total_upload_payload += r["payload_bytes"]
                    total_upload_protocol += r["protocol_bytes"]
                else:
                    failed_upload_bytes += r["total_bytes"]

        total_valid_bytes = (
            total_download_payload
            + total_download_protocol
            + total_upload_payload
            + total_upload_protocol
        )
        total_failed_bytes = failed_download_bytes + failed_upload_bytes

        return {
            "total_download_bytes": total_download_payload + total_download_protocol,
            "total_upload_bytes": total_upload_payload + total_upload_protocol,
            "total_download_payload_bytes": total_download_payload,
            "total_upload_payload_bytes": total_upload_payload,
            "total_protocol_bytes": total_download_protocol + total_upload_protocol,
            "total_failed_bytes": total_failed_bytes,
            "cumulative_total_bytes": total_valid_bytes + total_failed_bytes,
            "cumulative_total_mb": round((total_valid_bytes + total_failed_bytes) / (1024 * 1024), 4),
            "total_transmissions": len(self.records),
        }

    def get_round_summary(self, round_idx: int) -> Dict[str, Any]:
        """Calculates communication statistics for a specific round."""
        round_records = [r for r in self.records if r["round"] == round_idx]
        dl_bytes = sum(r["total_bytes"] for r in round_records if r["direction"] == "download" and r["success"])
        ul_bytes = sum(r["total_bytes"] for r in round_records if r["direction"] == "upload" and r["success"])
        failed_bytes = sum(r["total_bytes"] for r in round_records if not r["success"])

        return {
            "round": round_idx,
            "download_bytes": dl_bytes,
            "upload_bytes": ul_bytes,
            "failed_bytes": failed_bytes,
            "round_total_bytes": dl_bytes + ul_bytes + failed_bytes,
            "round_total_kb": round((dl_bytes + ul_bytes + failed_bytes) / 1024.0, 2),
            "transmission_count": len(round_records),
        }

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.records)
