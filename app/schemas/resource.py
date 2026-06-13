from pydantic import ConfigDict
from pydantic import BaseModel

from app.models.resource import ResourceType


class ResourceResponse(BaseModel):
    """Read-only resource view returned as part of a curriculum response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    type: ResourceType
    source_ref: str

    # raw_content is intentionally excluded:
    #   it is internal processing data (extracted text) and can be very large.
