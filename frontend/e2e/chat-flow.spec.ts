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

test("user reviews a generated plan and sends feedback through HIL resume", async ({ page }) => {
  await page.route("**/auth/me", async (route) => {
    await route.fulfill({ status: 401, contentType: "application/json", body: '{"detail":"unauthorized"}' })
  })
  await page.route("**/auth/login", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: '{"ok":true}' })
  })
  await page.route("**/stream", async (route) => {
    expect(route.request().postDataJSON()).toMatchObject({ query: "请帮我制定一周学习计划" })
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "cache-control": "no-cache" },
      body: [
        'data: {"type":"thread_id","thread_id":"plan-hil-e2e"}',
        "",
        'data: {"type":"interrupt","draft":"原始计划：周一复习数学，周三复习语文。","thread_id":"plan-hil-e2e"}',
        "",
        'data: {"type":"done"}',
        "",
      ].join("\n"),
    })
  })
  await page.route("**/resume", async (route) => {
    expect(route.request().method()).toBe("POST")
    expect(route.request().postDataJSON()).toEqual({
      thread_id: "plan-hil-e2e",
      feedback: "把周三改成物理，并保留每天两小时上限",
    })
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "cache-control": "no-cache" },
      body: [
        'data: {"type":"interrupt","draft":"修订计划：周一复习数学，周三复习物理；每天不超过两小时。","thread_id":"plan-hil-e2e"}',
        "",
        'data: {"type":"done"}',
        "",
      ].join("\n"),
    })
  })

  await page.goto("/")
  await page.getByLabel("账号").fill("admin")
  await page.getByLabel("密码").fill("123456")
  await page.getByRole("button", { name: "登录" }).click()

  const input = page.getByPlaceholder("输入你的问题...")
  await input.fill("请帮我制定一周学习计划")
  await input.press("Enter")

  await expect(page.getByText("学习计划草稿 — 请审阅")).toBeVisible()
  await expect(page.locator("textarea").nth(1)).toHaveValue("原始计划：周一复习数学，周三复习语文。")
  await page.getByPlaceholder("例如：把周三的数学改成物理").fill(
    "把周三改成物理，并保留每天两小时上限"
  )
  await page.getByRole("button", { name: "要求修改" }).click()

  await expect(page.getByText("修订计划：周一复习数学，周三复习物理；每天不超过两小时。")).toBeVisible()
  await expect(page.getByText("学习计划草稿 — 请审阅")).toBeVisible()
})
