const sidebar = document.getElementById("sidebar");
const mainContent = document.getElementById("mainContent");
const toggleButton = document.getElementById("toggleSidebar");
const STORAGE_KEY = "cygnusaSidebarCollapsed";

const applyState = (collapsed) => {
    if (collapsed) {
        sidebar.classList.add("collapsed");
        mainContent.classList.add("collapsed");
    } else {
        sidebar.classList.remove("collapsed");
        mainContent.classList.remove("collapsed");
    }
};

const storedState = localStorage.getItem(STORAGE_KEY);
applyState(storedState === "true");

if (toggleButton) {
    toggleButton.addEventListener("click", () => {
        const isCollapsed = sidebar.classList.contains("collapsed");
        applyState(!isCollapsed);
        localStorage.setItem(STORAGE_KEY, String(!isCollapsed));
    });
}

const globalRoleSelect = document.getElementById("globalRoleSelect");
if (globalRoleSelect) {
    globalRoleSelect.addEventListener("change", (event) => {
        const role = event.target.value;
        if (role) {
            fetch("/set-role", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ role }),
            }).then(() => window.location.reload());
        }
    });
}
