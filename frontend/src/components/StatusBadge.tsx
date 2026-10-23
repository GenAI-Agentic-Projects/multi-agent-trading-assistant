interface StatusBadgeProps {
  active: boolean;
}

export function StatusBadge({ active }: StatusBadgeProps) {
  return (
    <span className={`status-badge ${active ? "is-active" : "is-inactive"}`}>
      <span className="status-dot" aria-hidden="true" />
      {active ? "Active" : "Inactive"}
    </span>
  );
}