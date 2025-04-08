console.log("from script file");
// Script to handle dashboard filter actions
document.addEventListener("DOMContentLoaded", function () {
  // Reset filters button
  const resetButton = document.querySelector(".btn-outline-secondary");
  resetButton.addEventListener("click", function () {
    // Reset all input fields
    document.querySelectorAll("input").forEach((input) => {
      input.value = "";
    });

    // Reset all select fields
    document.querySelectorAll("select").forEach((select) => {
      select.selectedIndex = 0; // Reset to default
    });

    alert("Filters have been reset.");
  });

  // Apply filters button
  const applyButton = document.querySelector(".btn-primary");
  applyButton.addEventListener("click", function () {
    // Gather filter values
    const filters = {
      itemMrp: document.querySelector("input[placeholder='Enter MRP']").value,
      outletType: document.querySelector("select[aria-label='Outlet Type']").value,
      outletLocation: document.querySelector("input[placeholder='Enter Location']").value,
      outletSize: document.querySelector("select[aria-label='Outlet Size']").value,
      fatContent: document.querySelector("select[aria-label='Fat Content']").value,
      yearEstablished: document.querySelector("input[placeholder='Enter Year']").value,
    };

    console.log("Filters applied:", filters);
    alert("Filters applied! Check the console for details.");
    // Further processing logic (e.g., API call) can be added here
  });
});