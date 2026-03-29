from pydantic import BaseModel
from typing import Optional, List


class Vendor(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    tax_id: Optional[str] = None


class BillTo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    email: Optional[str] = None


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None

    vendor: Vendor
    bill_to: BillTo

    line_items: List[LineItem]

    subtotal: Optional[float] = None
    tax_rate: Optional[float] = None
    tax_amount: Optional[float] = None
    discount: Optional[float] = None
    total_amount: Optional[float] = None

    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    payment_method: Optional[str] = None
    notes: Optional[str] = None

    confidence_score: float