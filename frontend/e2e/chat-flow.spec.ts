import { expect, test } from "@playwright/test"

test("user logs in, sends a question, and receives an SSE answer", async ({ page }) => {
  await page.route("**/auth/me", async (route) => {
    await route.fulfill({ status: 401, contentType: "application/json", body: '{"detail":"unauthorized"}' })
  })
  await page.route("**/auth/login", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: '{"ok":true}' })
  })
  await page.route("**/stream", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.postDataJSON()).toMatchObject({ query: "二次函数怎么复习？" })
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "cache-control": "no-cache" },
      body: [
        'data: {"type":"thread_id","thread_id":"browser-e2e"}',
        "",
        'data: {"type":"token","content":"先梳理知识点，再做典型题。"}',
        "",
        'data: {"type":"done"}',
        "",
      ].join("\n"),
    })
  })

  await page.goto("/")
  await expect(page.getByRole("heading", { name: "高考辅导 AI 助手" })).toBeVisible()
  await page.getByLabel("账号").fill("admin")
  await page.getByLabel("密码").fill("123456")
  await page.getByRole("button", { name: "登录" }).click()

  const input = page.getByPlaceholder("输入你的问题...")
  await expect(input).toBeVisible()
  await input.fill("二次函数怎么复习？")
  await input.press("Enter")

  await expect(page.getByText("二次函数怎么复习？").last()).toBeVisible()
  await expect(page.getByText("先梳理知识点，再做典型题。")).toBeVisible()
})
