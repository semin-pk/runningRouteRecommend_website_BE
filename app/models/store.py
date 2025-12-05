from sqlalchemy import (
    CHAR,
    DECIMAL,
    VARCHAR,
    JSON,
    Index,
    UniqueConstraint,
    ForeignKey,
    Column,  # ✅ 추가
)
from sqlalchemy.orm import relationship

from app.models.base import Base


class StoreInfo(Base):
    """가게 정보 테이블 모델"""

    __tablename__ = "store_info"

    store_id = Column(CHAR(36), primary_key=True, nullable=False, comment="UUID (PK)")
    name = Column(VARCHAR(255), nullable=False, comment="가게 이름")
    address = Column(VARCHAR(255), nullable=False, comment="상세 주소")
    longitude = Column(DECIMAL(10, 7), nullable=False, comment="경도")
    latitude = Column(DECIMAL(10, 7), nullable=False, comment="위도")
    phone = Column(VARCHAR(30), nullable=True, comment="전화번호")
    open_time = Column(VARCHAR(10), nullable=True, comment="영업 시작 시간 (예: '10:00')")
    close_time = Column(VARCHAR(10), nullable=True, comment="영업 종료 시간 (예: '22:00')")

    # Relationships
    review_summary = relationship(
        "StoreReviewSummary",
        back_populates="store",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Indexes and Constraints
    __table_args__ = (
        UniqueConstraint(
            "name", "address", "longitude", "latitude", name="uniq_store_physical"
        ),
        Index("idx_store_name", "name"),
        Index("idx_store_address", "address"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    def __repr__(self):
        return f"<StoreInfo(store_id={self.store_id}, name={self.name})>"


class StoreReviewSummary(Base):
    """가게 리뷰 요약 테이블 모델"""

    __tablename__ = "store_review_summary"

    store_id = Column(
        CHAR(36),
        ForeignKey("store_info.store_id", ondelete="CASCADE", onupdate="CASCADE"),
        primary_key=True,
        nullable=False,
        comment="PK + FK",
    )
    store_name = Column(
        VARCHAR(255),
        nullable=False,
        comment="가게 이름 (조회 편하게 중복 저장)",
    )
    main_menu = Column(JSON, nullable=False, comment="대표 메뉴 리스트")
    atmosphere = Column(JSON, nullable=False, comment="분위기/경험")
    recommended_for = Column(JSON, nullable=False, comment="추천 대상")

    # Relationships
    store = relationship("StoreInfo", back_populates="review_summary")

    __table_args__ = (
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    def __repr__(self):
        return f"<StoreReviewSummary(store_id={self.store_id}, store_name={self.store_name})>"
