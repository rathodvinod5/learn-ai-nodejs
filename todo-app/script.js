document.getElementById("add-btn").addEventListener("click", function() {
    const input = document.getElementById("todo-input");
    const taskText = input.value;
    if (taskText === "") {
        alert("Please enter a task.");
        return;
    }
    const listItem = document.createElement("li");
    listItem.textContent = taskText;
    document.getElementById("todo-list").appendChild(listItem);
    input.value = "";
});
